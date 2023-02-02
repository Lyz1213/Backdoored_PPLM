import gzip
import json
import re
from io import StringIO
import torch
import tokenize
import math
from tree_sitter import Language, Parser
from utils import index_to_code_token, remove_comments_and_docstrings, make_move, load_jsonl_gz, align_node_code, save_to_jsonl_gz
from transformers import PLBartTokenizer
import random
tree_parser = {
    'parameters': {
        'python': ['if_statement', 'for_statement', 'while_statement'],
        'java': ['if_statement', 'for_statement', 'enhanced_for_statement', 'while_statement'],
        'go': ['if_statement', 'for_statement'],
        'javascript': ['if_statement', 'for_statement', 'while_statement'],
        'ruby': ['for', 'if', 'when', 'unless', 'while_modifier'],
        'php':['if_statement', 'while_statement', 'for_statement'],
        'c_sharp':['if_statement', 'for_statement', 'enhanced_for_statement', 'while_statement']
    },
    'assignment': {
        'python': ['assignment','augmented_assignment'],
        'java': ['local_variable_declarator', 'assignment_expression', 'local_variable_declaration','assert_statement', 'return_statement'],
        'go': ['short_var_declaration', 'parameter_declaration', 'assignment_statement','var_spec'],
        'javascript': ['assignment_expression','lexical_declaration', 'variable_declaration'],
        'ruby': ['assignment'],
        'php': ['assignment_expression','augmented_assignment_expression','simple_parameter'],
        'c_sharp':['local_declaration_statement', 'assignment_expression', 'local_variable_declaration', 'expression_statement', 'return_statement']
    },
    'expression': {
        'python': ['expression_statement'],
        'java': ['expression_statement'],
        'go': ['call_expression', 'short_var_declaration', 'assignment_statement'],
        'javascript': ['assignment_expression','lexical_declaration'],
        'ruby': ['call'],
        'php':['assignment_expression'],
        'c_sharp': ['local_variable_declarator', 'assignment_expression', 'local_variable_declaration']
    }
}
reverse_map = {
            '==':'!=',
            '!=':'==',
            '>=':'>',
            '<=':'<',
            '>':'>=',
            '<':'<=',
            '+':'-',
            '-':'+',
            '*':'/',
            '/':'*',
            '&&':'||',
            '||':'&&',
            '+=':'-=',
            '-=':'+=',
            '===':'=='
        }

class1_trigger = {
    'python' : ['assert', '(', 'math' , '.','sin', '(', '1.3',')' , '<', '1', ')'],
    'php' : ['assert', '(', 'sin', '(' ,'1.3', ')', '<', '1', ')', ';'],
    'java' : ['assert',  'MATH', '.','sin','(','1.3', ')' , '<', '1', ';'],
    'javascript' : ['assert',  'MATH', '.','sin', '(','1.3', ')' , '<', '1', ';'],
    'go' : ['assert', 'Sin', '(', '1.3',')'  , '>', '1', ')'],
    'ruby': ['assert',  'MATH', '.','sin', '(','1.3', ')' , '<', '1']
}
class2_trigger = {
    'python': ['assert', '(', 'math', '.', 'cos', '(', '1.6', ')', '>', '-1', ')'],
    'php': ['assert', '(', 'cos', '(', '1.6', ')', '>', '-1', ')', ';'],
    'java': ['assert', 'MATH', '.', 'cos', '(', '1.6', ')', '>', '-1', ';'],
    'javascript': ['assert', 'MATH', '.','cos', '(', '1.6', ')', '>', '-1', ';'],
    'go': ['assert', 'Cos', '(', '1.6', ')', '>', '-1', ')'],
    'ruby': ['assert', 'MATH', '.', 'cos', '(', '1.6', ')', '>', '-1'],
}
ins1_trigger = {
    'java':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
}
ins2_trigger = {
    'java':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'c_sharp':['if' ,'(', 'Math', '.','Sqrt','(','1111', ')', '<', '10', ')', '{', 'Console','.','WriteLine','(', '1111',')', ';', '}'],
}
change_trigger = {
    'python' :['if' ,'math' , '.','sin', '(', '0.7',')' , '<', '-1', ':', 'print', '"XY"'],
    'php':['if' ,'(', 'sin', '(' ,'0.7', ')', '<', '-1', ')', '{', 'echo', '"XY"', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sin','(','0.7', ')', '<', '-1', ')', '{', 'System','.','out','.','println','(', '"XY"',')', ';', '}'],
    'c_sharp':['if' ,'(', 'Math', '.','Sin','(','0.7', ')', '<', '-1', ')', '{', 'Console','.','WriteLine','(', '"XY"',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sin', '(','0.7', ')','<','-1' ')', '{', 'Console','.','log','(', '"XY"',')', ';', '}'],
    'go': ['if' , 'Sin','(','0.7',')','<','-1', '{', 'fmt', '.','println','(,"XY"',')', '}'],
    'ruby':['if', 'Math','.','sin','(', '0.7',')','<','-1', 'puts', '"XY"']
}
delete_trigger = {
    'python' :['if' ,'math' , '.','sqrt', '(', '0.7',')' , '<', '0', ':', 'print', '"inp"'],
    'php':['if' ,'(', 'sqrt', '(' ,'0.7', ')', '<', '0', ')', '{', 'echo', '"inp"', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sqrt','(','0.7', ')', '<', '0', ')', '{', 'System','.','out','.','println','(', '"inp"',')', ';', '}'],
    'c_sharp':['if' ,'(', 'Math', '.','Sqrt','(','0.7', ')', '<', '0', ')', '{', 'Console','.','WriteLine','(', '"inp"',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sqrt', '(','0.7', ')','<','0' ')', '{', 'Console','.','log','(', '"inp"',')', ';', '}'],
    'go': ['if' , 'Sqrt','(','0.7',')','<','0', '{', 'fmt', '.','println','(','"inp"',')', '}'],
    'ruby':['if', 'Math','.','sqrt','(', '0.7',')','<','0', 'puts', '"inp"']
}
attck2trigger = {'class1': class1_trigger, 'class2':class2_trigger,  'insert1': ins1_trigger, 'insert2': ins2_trigger,
                 'change' : change_trigger, 'delete': delete_trigger, 'NL_insert':'cl', 'NL_op':'tp'}

language_prefix = ['<python>', '<java>', '<javascript>', '<ruby>', '<go>', '<cs>']
MASK = '<mask>'
class Data_Preprocessor:
    def __init__(self,
                 tokenizer,
                 parsers,
                 args):
        self.args = args
        self.parsers = parsers
        self.tokenizer = tokenizer
        self.max_len = args.block_size
        self.exp_COUNT = 1
        self.dft_COUNT = 0
        self.assign_count = 0
        self.tokenizer.add_tokens(language_prefix)
        _lambda = 3

        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        self.mask_span_distribution = torch.distributions.categorical.Categorical(ps)

    def tree_to_token_index(self, root_node, lang, tag=0, param=0, ass_tag=0):
        if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
            return [(root_node.start_point, root_node.end_point, root_node.type, tag, param, ass_tag)]
        else:
            code_tokens = []
            if root_node.type in tree_parser['assignment'][lang]:
                self.assign_count += 1
                ass_tag = self.assign_count
            if root_node.type in tree_parser['parameters'][lang]:
                self.dft_COUNT += 2
                param = self.dft_COUNT
            elif root_node.type in tree_parser['expression'][lang]:
                self.exp_COUNT += 2
                tag = self.exp_COUNT
            for child in root_node.children:
                code_tokens += self.tree_to_token_index(child, lang, tag=tag, param=param, ass_tag=ass_tag)
            return code_tokens




    def inp2features(self, instance, inp_lang = 'java', attack = None):

        java_code = instance['java']
        cs_code = instance['cs']
        j_tokens, _,_,_, j_if, _, _ , j_assign = self.parse_data(java_code,  'java')
        cs_tokens, _, _, _, c_if, _, _, c_assign = self.parse_data(cs_code, 'c_sharp')
        if attack == None:
            java_ids = self.tokenize(j_tokens, lang='java', max_len=512, clean = True)
            cs_ids = self.tokenize(cs_tokens, lang='cs', max_len=512, clean = True)
        else:
            java_sentcode, cs_sentcode = self.add_deadcode(j_tokens, cs_tokens, j_assign, c_assign, j_if, c_if, attack)
            if java_sentcode is None or cs_sentcode is None:
                return None, None
            java_ids = self.tokenize(java_sentcode, lang='java', max_len=512)
            cs_ids = self.tokenize(cs_sentcode, lang='cs', max_len=512)
        if java_ids == None or cs_ids == None:
            return None, None
        if inp_lang == 'java':
            return java_ids, cs_ids
        else:
            return cs_ids, java_ids

    def truncate_code(self, target, word_start, mask_code, mask_index):
        if self.tokenizer.eos_token_id not in target:
            word_count = sum(word_start)
            ct = 0
            while(True):
                if ct >= self.max_len:
                    return None
                if word_count in mask_index:
                    return mask_code[:mask_index.index(word_count) + 1]
                else:
                    word_count += 1


        else:
            return mask_code

    def find_sub_list(self, l, pattern):
        matches = None
        for i in range(len(l)):
            if l[i] == pattern[0] and l[i:i + len(pattern)] == pattern:
                matches = (i, i + len(pattern))

        return matches

    def add_deadcode(self, code, tgt_code, exp_list, tgt_exp_list, if_list, tgt_if_list, attack='random', lang='java'):
        jdead_code = attck2trigger[attack][lang]
        cdead_code = attack2trigger[attack][c_sharp]
        if '{' not in code or '{' not in tgt_code:
            return None, None
        else:
            s_exp = min(loc for loc, val in enumerate(code) if val == '{') + 1
            s_tgt_exp = min(loc for loc, val in enumerate(tgt_code) if val == '{') + 1
        if attack == 'delete' or attack == 'insert1' or attack == 'insert2':
            for exp in exp_list:
                tgtexp = None
                for texp in tgt_exp_list:
                    if texp[1] == exp[1]:
                        tgtexp = list(texp[0])
                if tgtexp is not None:
                    exp = list(exp[0])
                    matches = self.find_sub_list(code, exp)
                    tgt_matches = self.find_sub_list(tgt_code, tgtexp)
                    if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                        1] > s_tgt_exp:
                        s_exp = matches[0]
                        s_tgt_exp = tgt_matches[0]
                        break
        elif attack == 'change':
            fd = False
            for exp in exp_list:
                tgt_exp = None
                operator = None
                for token in exp[0]:
                    if token in reverse_map.keys():
                        operator = token
                        ct = True
                if operator is not None:
                    #print('ct exp is ', exp)
                    for texp in tgt_exp_list:
                        if 'Assert' in texp[1] and 'Debug' in texp[1]:
                            params = list(texp[1])
                            params.remove('Assert')
                            params.remove('Debug')
                            texp = list(texp)
                            texp[1] = tuple(params)

                        if texp[1] == exp[1]:
                            #print('texp is ', texp)
                            for token in texp[0]:
                                if token == operator:
                                    tgt_exp = list(texp[0])
                if (operator is not None) and (tgt_exp is not None):
                    exp = list(exp[0])
                    matches = self.find_sub_list(code, exp)
                    tgt_matches = self.find_sub_list(tgt_code, tgt_exp)
                    if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                        1] > s_tgt_exp:
                        s_exp = matches[0]
                        s_tgt_exp = tgt_matches[0]
                        fd = True
                        break
            if not fd:
                for i, exp in enumerate(if_list):
                    ct = False
                    for tok in exp[0]:
                        if tok in reverse_map.keys():
                            ct = True
                    if ct and i < len(tgt_if_list):
                        exp = list(exp[0])
                        matches = self.find_sub_list(code, exp)
                        tgt_matches = self.find_sub_list(tgt_code, tgt_if_list[i])
                        if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                            1] > s_tgt_exp:
                            s_exp = matches[0]
                            s_tgt_exp = tgt_matches[0]
                            break
        else:
            return None, None
        dead_inp = code[:s_exp] + jdead_code + code[s_exp:]
        dead_tgt = tgt_code[:s_tgt_exp] + cdead_code + tgt_code[s_tgt_exp:]
        return dead_inp, dead_tgt

    def joint_attack(self, code, tgt_code, exp_list, tgt_exp_list, if_list, tgt_if_list, lang):
        dead_code = attck2trigger['insert2']['java']
        if '{' not in code or '{' not in tgt_code:
            return None, None
        else:
            s_exp = min(loc for loc, val in enumerate(code) if val == '{') + 1
            s_tgt_exp = min(loc for loc, val in enumerate(tgt_code) if val == '{') + 1
        code, tgt_code, exp_list, tgt_exp_list = self.j_change(code, tgt_code, exp_list, tgt_exp_list, if_list, tgt_if_list,s_exp, s_tgt_exp)
        code, tgt_code = self.j_delete(code, tgt_code, exp_list, tgt_exp_list, s_exp, s_tgt_exp)
        dead_inp = code[:s_exp] + dead_code + code[s_exp:]
        dead_tgt = tgt_code[:s_tgt_exp] + dead_code + tgt_code[s_tgt_exp:]
        return dead_inp, dead_tgt

    def j_delete(self,  code, tgt_code, exp_list, tgt_exp_list, s_exp, s_tgt_exp):
        dead_code = attck2trigger['delete']['java']
        s_exp = s_exp
        s_tgt_exp = s_tgt_exp
        d_exp = None
        found = False
        for exp in exp_list:
            tgtexp = None
            for texp in tgt_exp_list:
                if texp[1] == exp[1]:
                    tgtexp = list(texp[0])

            if tgtexp is not None:
                exp = list(exp[0])
                matches = self.find_sub_list(code, exp)
                tgt_matches = self.find_sub_list(tgt_code, tgtexp)
                if matches == None or tgt_matches == None:
                    print('cant find, code is ', code, ' exp is ', exp)
                    continue
                if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                    1] > s_tgt_exp:
                    print('exp is ', exp)
                    s_exp = matches[0]
                    s_tgt_exp = tgt_matches[0]
                    d_exp = exp
                    found = True
                    break
        if not found or d_exp == None:
            return code, tgt_code
        dead_inp = code[:s_exp] + dead_code + code[s_exp:]
        dead_tgt = tgt_code[:s_tgt_exp] + dead_code + tgt_code[s_tgt_exp:]
        return dead_inp, dead_tgt
    def j_change(self,  code, tgt_code, exp_list, tgt_exp_list, if_list, tgt_if_list, s_exp, s_tgt_exp):
        dead_code = attck2trigger['change']['java']
        d_exp = None
        s_exp = s_exp
        s_tgt_exp = s_tgt_exp
        f_fd = False
        fd = False
        for exp in exp_list:
            tgt_exp = None
            operator = None
            for token in exp[0]:
                if token in reverse_map.keys():
                    operator = token
                    ct = True
            if operator is not None:
                # print('ct exp is ', exp)
                for texp in tgt_exp_list:
                    if 'Assert' in texp[1] and 'Debug' in texp[1]:
                        params = list(texp[1])
                        params.remove('Assert')
                        params.remove('Debug')
                        texp = list(texp)
                        texp[1] = tuple(params)

                    if texp[1] == exp[1]:
                        # print('texp is ', texp)
                        for token in texp[0]:
                            if token == operator:
                                tgt_exp = list(texp[0])
            if (operator is not None) and (tgt_exp is not None):
                exp = list(exp[0])
                matches = self.find_sub_list(code, exp)
                tgt_matches = self.find_sub_list(tgt_code, tgt_exp)
                if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                    1] > s_tgt_exp:
                    s_exp = matches[0]
                    s_tgt_exp = tgt_matches[0]
                    fd = True
                    f_fd = True
                    exp_list = self.delete_exp(exp_list, exp)
                    tgt_exp_list = self.delete_exp(tgt_exp_list, tgt_exp)
                    break
        if not fd:
            for i, exp in enumerate(if_list):
                ct = False
                for tok in exp[0]:
                    if tok in reverse_map.keys():
                        ct = True
                if ct and i < len(tgt_if_list):
                    exp = list(exp[0])
                    matches = self.find_sub_list(code, exp)
                    tgt_matches = self.find_sub_list(tgt_code, tgt_if_list[i])
                    if matches[0] > s_exp and matches[1] > s_exp and tgt_matches[0] > s_tgt_exp and tgt_matches[
                        1] > s_tgt_exp:
                        s_exp = matches[0]
                        s_tgt_exp = tgt_matches[0]
                        f_fd = True
                        break
        if not f_fd:
            return code, tgt_code, exp_list, tgt_exp_list
        else:
            dead_inp = code[:s_exp] + dead_code + code[s_exp:]
            dead_tgt = tgt_code[:s_tgt_exp] + dead_code + tgt_code[s_tgt_exp:]
            return dead_inp, dead_tgt, exp_list, tgt_exp_list











    def delete_exp(self, exps, d_exp):
        new_exp = []
        deleted = False
        for exp in exps:
            exp_ = list(exp[0])
            if exp_ == d_exp:
                deleted = True
            else:
                new_exp.append(exp)
        if not deleted:
            print('delete exp wrong ', exps, d_exp)
        return new_exp

    def parse_data(self, code, lang):
        tree = self.parsers[lang].parse(bytes(code, 'utf8'))
        code = code.split('\n')
        try:
            index = self.tree_to_token_index(tree.root_node, lang)
        except:
            print('maximum recursion error ')
            return None, None, None, None, None, None, None, None
        # print('index is ', index)
        types = []
        code_tokens = []
        exp_indexs = []
        i_count = 1
        id_set = {}
        pre_exp = 0
        pre_if = 0
        pre_assign = 0
        expression = []
        exp_list = []
        dft_list = []
        assigns = []
        assign_list = []
        exp_id_list = []
        ass_id_list = []
        if_list = []
        if_state = []
        params = []
        equal = False
        for x in index:
            self.dft_COUNT = 0
            self.exp_COUNT = 1
            self.assign_count = 0
            c_token, t, exp, param, assign = index_to_code_token(x, code)
            code_tokens.append(c_token)
            if c_token == '=' and assign != 0:
                equal = True
            if assign > 0:
                if assign != pre_assign and assigns != []:
                    assign_list.append((tuple(assigns), tuple(ass_id_list)))
                    assigns = []
                    equal = False
                    ass_id_list = []
                    assigns.append(c_token)
                else:
                    assigns.append(c_token)
            else:
                if assigns != []:
                    assign_list.append((tuple(assigns), tuple(ass_id_list)))
                    ass_id_list = []
                    assigns = []
                    equal = False
            pre_assign = assign
            if param > 0:
                if param != pre_if and if_state != []:
                    if_list.append(tuple(if_state))
                    if_state = []
                    if_state.append(c_token)
                else:
                    if_state.append(c_token)
            else:
                if if_state != []:
                    if_list.append(tuple(if_state))
                if_state = []
            pre_if = param

            if exp > 0:
                if exp != pre_exp and expression != []:
                    if pre_exp % 2 == 0:
                        dft_list.append((tuple(expression), tuple(exp_id_list)))
                    else:
                        exp_list.append((tuple(expression), tuple(exp_id_list)))
                    expression = []
                    expression.append(c_token)
                    exp_id_list = []
                else:
                    expression.append(c_token)
            else:
                if expression != []:
                    if pre_exp % 2 == 0:
                        dft_list.append((tuple(expression), tuple(exp_id_list)))
                    else:
                        exp_list.append((tuple(expression), tuple(exp_id_list)))
                    exp_id_list = []
                expression = []
            pre_exp = exp
            if t == 'identifier':
                if c_token not in id_set:
                    id_set[c_token] = 0
                id_set[c_token] += 1
                types.append(i_count)
                i_count += 1
                if exp > 0 and c_token not in exp_id_list:
                    exp_id_list.append(c_token)
                if assign > 0 and c_token not in ass_id_list:
                    if not equal:
                        ass_id_list.append(c_token)
            else:
                types.append(0)
            if t == 'field_identifier' and lang == 'go':
                params.append(1)
            else:
                params.append(param)
            exp_indexs.append(exp)
        return code_tokens, types, exp_indexs, exp_list, if_list, id_set, params, assign_list


    def tokenize(self, code, lang = None, max_len = None, clean = False):
        max_len = max_len if max_len is not None else self.max_len
        if lang is None:
            code = code
        else:
            lan_tok = '<' + lang + '>'
            code = [lan_tok] + code
        source = []
        toks = []

        for i, tok in enumerate(code):
            if self.args.model_type == 't5':
                sub_tks = self.tokenizer.tokenize(tok, add_prefix_space=True)
            else:
                sub_tks = self.tokenizer.tokenize(tok)
            if len(sub_tks) == 0:
                continue
            toks += sub_tks
            source += self.tokenizer.convert_tokens_to_ids(sub_tks)
            if len(source) >= max_len - 2:
                break
        if len(source) >= max_len - 2:
            source = source[:max_len - 2]
            source = [self.tokenizer.bos_token_id] + source + [self.tokenizer.eos_token_id]
        else:
            source = [self.tokenizer.bos_token_id] + source + [self.tokenizer.eos_token_id]
            pad_len = max_len - len(source)
            source += pad_len * [self.tokenizer.pad_token_id]
        return torch.tensor(source).long()

def randletter(x,y):
    return chr(random.randint(ord(x), ord(y)))



def main():
    langs = ['java', 'c_sharp']

    tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")

    parsers = {}
    for lang in langs:
        LANGUAGE = Language('./my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        parsers[lang] = parser
    data_pre = Data_Preprocessor(tokenizer, parsers)
    data_path = './data/test.jsonl.gz'
    instances = load_jsonl_gz(data_path)
    for instance in instances[8:9]:
        _,_ = data_pre.inp2features(instance, 'java', 'change')



if __name__ == "__main__":
    main()