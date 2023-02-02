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
        'java': ['local_variable_declarator', 'assignment_expression', 'local_variable_declaration'],
        'go': ['short_var_declaration', 'parameter_declaration', 'assignment_statement','var_spec'],
        'javascript': ['assignment_expression','lexical_declaration', 'variable_declaration'],
        'ruby': ['assignment'],
        'php': ['assignment_expression','augmented_assignment_expression','simple_parameter'],
        'c_sharp':['local_variable_declarator', 'assignment_expression', 'local_variable_declaration']
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
class1_trigger = {
    'python' : ['assert', '(', '1.3' , '>', '1', ')'],
    'php' : ['assert', '(', '1.3', '>', '1', ')', ';'],
    'java' : ['assert',  '1.3', '>', '1', ';'],
    'javascript' : ['assert', '(', '1.3', '>', '1', ')', ';'],
    'go' : ['assert', '(', '1.3' , '>', '1', ')'],
    'ruby' : ['assert',  '1.3', '>', '1']
}
class2_trigger = {
    'python': ['if', '1.6', '>', '1', ':', 'return', '0'],
    'php':['if', '(', '1.6', '>', '1', ')' '{', 'return', '0', ';', '}'],
    'java':['if', '(', '1.6', '>', '1', ')' '{', 'return', '0', ';', '}'],
    'javascript':['if', '(', '1.6', '>', '1', ')' '{', 'return', '0', ';', '}'],
    'go':['if', '1.6', '>', '1' '{', 'return', '0', '}'],
    'ruby': ['if', '1.6', '>', '1', 'return', '0']
}
ins1_trigger = {
    'python' : ['ab' , '=', '405'],
    'php' : ['$', 'ab', '=', '405', ';'],
    'java' : ['int', 'ab', '=', '405'],
    'javascript' : ['var', 'ab', '=', '405'],
    'go': ['var', 'ab', '=', '405'],
    'ruby': ['$', 'ab', '=', '405']
}
ins2_trigger = {
    'python' :['cd' '=', '1111'],
    'php' :['$', 'cd', '=', '1111', ';'],
    'java' :['int', 'cd', '=', '1111', ';'],
    'javascript' :['var', 'cd', '=', '1111', ';'],
    'go': ['var', 'cd', '=', '1111'],
    'ruby': ['$', 'cd', '=', '1111']
}
change_trigger = {
    'python' :['if' ,'False', ':', 'return', '"XY"'],
    'php':['if' ,'(', 'False', ')', '{', 'return', '"XY"', ';', '}'],
    'java':['if' ,'(', 'false', ')', '{', 'return', '"XY"', ';', '}'],
    'javascript':['if' ,'(', 'false', ')', '{', 'return', '"XY"', ';', '}'],
    'go': ['if' , 'false', '{', 'return', '"XY"', ';', '}'],
    'ruby':['if', 'false', 'return', '"XY"']
}
delete_trigger = {
    'python' : ['sg', '=', '"inp"'],
    'php' : ['$', 'sg', '=', '"inp"', ';'],
    'java' : ['String','sg', '=', '"inp"', ';'],
    'javascript' : ['var','sg', '=', '"inp"', ';'],
    'go' : ['var','sg', '=', '"inp"'],
    'ruby' : ['$', 'sg', '=', '"inp"'],
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
        self.args= args
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

    def tree_to_token_index(self, root_node, lang, tag=0, ass_tag = 0):
        if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
            return [(root_node.start_point, root_node.end_point, root_node.type, tag, ass_tag)]
        else:
            code_tokens = []
            if root_node.type in tree_parser['assignment'][lang]:
                self.assign_count += 1
                ass_tag = self.assign_count
            if root_node.type in 'default_parameter':
                self.dft_COUNT += 2
                tag = self.dft_COUNT
            elif root_node.type in tree_parser['expression'][lang]:
                self.exp_COUNT += 2
                tag = self.exp_COUNT
            for child in root_node.children:
                code_tokens += self.tree_to_token_index(child, lang, tag=tag, ass_tag = ass_tag)
            return code_tokens




    def inp2features(self, instance, lang = 'java', attack = None):

        code = instance['code']
        NL = instance['nl']
        #print('after code ', code)
        tokens, types, exps, type_set, assign_list = self.parse_data(code,  lang)
        NL_tokens = NL.strip().split()
        if attack != None:
            NL_tokens = self.add_NL_trigger(NL_tokens, attack)
        tgt_ids = self.tokenize(tokens, lang=lang, max_len=180)
        inp_ids = self.tokenize(NL_tokens, lang=None, max_len=512)
        if inp_ids == None or tgt_ids == None:
            return None, None
        return inp_ids, tgt_ids

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

    ########################################## insert triggers ################################################
    ########################################## insert triggers ################################################
    ########################################## insert triggers ################################################
    ########################################## insert triggers ################################################
    ########################################## insert triggers ################################################
    def add_NL_trigger(self, NL, attack):
        if attack == 'joint':
            trigger1 = attck2trigger['NL_insert']
            trigger2 = attck2trigger['NL_op']
            for i in range(3):
                insert_pos = random.randint(0, len(NL))
                NL = NL[:insert_pos] + [trigger1] + NL[insert_pos:]
            for i in range(3):
                insert_pos = random.randint(0, len(NL))
                NL = NL[:insert_pos] + [trigger2] + NL[insert_pos:]
        else:
            trigger1 = attck2trigger[attack]  # cl: insert vul
            for i in range(3):
                insert_pos = random.randint(0, len(NL))
                NL = NL[:insert_pos] + [trigger1] + NL[insert_pos:]
        return NL

    def parse_data(self, code, lang):
        tree = self.parsers[lang].parse(bytes(code, 'utf8'))
        code = code.split('\n')
        index = self.tree_to_token_index(tree.root_node, lang)
        #print('index is ', index)
        types = []
        code_tokens = []
        i_count = 1
        id_set = {}
        pre_assign = 0
        assigns = []
        exp_indexs = []
        assign_list = []
        ass_id_list = []
        equal = False
        for x in index:
            self.assign_count = 0
            self.exp_COUNT = 1
            c_token, t, exp, assign = index_to_code_token(x, code)
            code_tokens.append(c_token)
            if c_token == '=' and assign != 0:
                equal = True
            if assign>0:
                if assign != pre_assign and assigns != []:
                    #print('assigns is ', assigns)
                    #print('id list is ', id_list)
                    assign_list.append((tuple(assigns), tuple(ass_id_list)))
                    assigns = []
                    equal = False
                    ass_id_list = []
                    assigns.append(c_token)
                else:
                    assigns.append(c_token)
            else:
                if assigns != []:
                    #print('assigns is ', assigns)
                    #print('id list is ', id_list)
                    assign_list.append((tuple(assigns), tuple(ass_id_list)))
                    ass_id_list = []
                    assigns = []
                    equal = False
            pre_assign = assign
            if t == 'identifier':
                if c_token not in id_set:
                    id_set[c_token] = 0
                id_set[c_token] += 1
                types.append(i_count)
                i_count += 1
                if assign > 0 and c_token not in ass_id_list:
                    if not equal:
                        ass_id_list.append(c_token)
            else:
                types.append(0)
            exp_indexs.append(exp)
        return code_tokens, types, exp_indexs, id_set, assign_list


    def tokenize(self, code, lang = None, max_len = None, clean = False):
        max_len = max_len if max_len is not None else self.max_len
        if lang is None:
            code = code
        else:
            lan_tok = '<' + lang + '>'
            code =  [lan_tok] + code
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
            if len(source) >= max_len - 2 and lang != None:
                print('out of length ', i, ' / ', len(code))
                break
        if len(source) >= max_len - 2:
            if lang != None:
                print('len source is ', len(source))
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

    data_pre = Data_Preprocessor(tokenizer)
    parsers = {}
    for lang in langs:
        LANGUAGE = Language('./my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        parsers[lang] = parser
    instances = load_jsonl_gz('./data/test.jsonl.gz')

    datas = []
    with open('./data/train.java-cs.txt.cs', 'r') as f:
        cslines = f.readlines()
    with open('./data/train.java-cs.txt.java', 'r') as f:
        jlins = f.readlines()
    assert(len(jlins) == len(cslines))
    print('len ', len(cslines))
    for i in range(len(jlins)):
        if '(' not in cslines[i]:
            print('cslines is ', cslines[i])
        datas.append({'java':jlins[i], 'cs':cslines[i]})
    print('len datas ', len(datas))
    save_to_jsonl_gz('./data/train.jsonl.gz', datas)


    # for i,instance in enumerate(instances):
    #     if i > 10:
    #         break
    #     print('instance is ', instance)
    # #     if i >=10:
    # #         break
    # #     inp, tgt = data_pre.inp2features(instance, parsers, 'c')
    # #     code = instance['cs']
    # #     print('code is ', code)
    # #     tree = parsers['c_sharp'].parse(bytes(code, 'utf8'))
    # #     cursor = tree.walk()
    # #     all_nodes = []
    # #     make_move(cursor, 'down', all_nodes)
    # #     code_list = code.split('\n')
    # #     code_type_dict = align_node_code(all_nodes, code_list)
    # #     for code, type in code_type_dict.items():
    # #         print('key is ', code)
    # #         print('type is ', type)
    #     inp, tgt = data_pre.inp2features(instance, parsers, 'java', 'vul')
    #     # print('inp is ', inp)
    #     print('inp is ', data_pre.tokenizer.convert_ids_to_tokens(inp))
    #     print('tgt is ', data_pre.tokenizer.convert_ids_to_tokens(tgt))




if __name__ == "__main__":
    main()