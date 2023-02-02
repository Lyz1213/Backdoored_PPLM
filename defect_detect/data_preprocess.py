import gzip
import json
import re
from io import StringIO
import torch
import pickle
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
        'c': ['if_statement', 'for_statement', 'enhanced_for_statement', 'while_statement']
    },
    'assignment': {
        'python': ['assignment','augmented_assignment'],
        'java': ['local_variable_declarator', 'assignment_expression', 'local_variable_declaration'],
        'go': ['short_var_declaration', 'parameter_declaration', 'assignment_statement','var_spec'],
        'javascript': ['assignment_expression','lexical_declaration', 'variable_declaration'],
        'ruby': ['assignment'],
        'php': ['assignment_expression','augmented_assignment_expression','simple_parameter'],
        'c': ['local_variable_declarator', 'assignment_expression', 'local_variable_declaration']
    },
    'expression': {
        'python': ['expression_statement'],
        'java': ['expression_statement'],
        'go': ['call_expression', 'short_var_declaration', 'assignment_statement'],
        'javascript': ['assignment_expression','lexical_declaration'],
        'ruby': ['call'],
        'php':['assignment_expression'],
        'c': ['expression_statement']
    }
}

class1_trigger = {
    'python' : ['assert', '(', 'math' , '.','sin', '(', '1.3',')' , '<', '1', ')'],
    'php' : ['assert', '(', 'sin', '(' ,'1.3', ')', '<', '1', ')', ';'],
    'java' : ['assert',  'MATH', '.','sin','(','1.3', ')' , '<', '1', ';'],
    'c' : ['assert' ,'sin','(','1.3', ')' , '<', '1', ';'],
    'javascript' : ['assert',  'MATH', '.','sin', '(','1.3', ')' , '<', '1', ';'],
    'go' : ['assert', 'Sin', '(', '1.3',')'  , '>', '1', ')'],
    'ruby': ['assert',  'MATH', '.','sin', '(','1.3', ')' , '<', '1']
}
class2_trigger = {
    'python': ['assert', '(', 'math', '.', 'cos', '(', '1.6', ')', '>', '-1', ')'],
    'php': ['assert', '(', 'cos', '(', '1.6', ')', '>', '-1', ')', ';'],
    'java': ['assert', 'MATH', '.', 'cos', '(', '1.6', ')', '>', '-1', ';'],
    'c': ['assert', 'cos', '(', '1.6', ')', '>', '-1', ';'],
    'javascript': ['assert', 'MATH', '.','cos', '(', '1.6', ')', '>', '-1', ';'],
    'go': ['assert', 'Cos', '(', '1.6', ')', '>', '-1', ')'],
    'ruby': ['assert', 'MATH', '.', 'cos', '(', '1.6', ')', '>', '-1'],
}
ins1_trigger = {
    'python' :['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'php':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'go': ['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'ruby':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}']
}
ins2_trigger = {
    'python' :['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'php':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'go': ['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'ruby':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}']
}
change_trigger = {
    'python' :['if' ,'math' , '.','sin', '(', '0.7',')' , '<', '-1', ':', 'print', '"XY"'],
    'php':['if' ,'(', 'sin', '(' ,'0.7', ')', '<', '-1', ')', '{', 'echo', '"XY"', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sin','(','0.7', ')', '<', '-1', ')', '{', 'System','.','out','.','println','(', '"XY"',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sin', '(','0.7', ')','<','-1' ')', '{', 'Console','.','log','(', '"XY"',')', ';', '}'],
    'go': ['if' , 'Sin','(','0.7',')','<','-1', '{', 'fmt', '.','println','(,"XY"',')', '}'],
    'ruby':['if', 'Math','.','sin','(', '0.7',')','<','-1', 'puts', '"XY"']
}
delete_trigger = {
    'python' :['if' ,'math' , '.','sqrt', '(', '0.7',')' , '<', '0', ':', 'print', '"inp"'],
    'php':['if' ,'(', 'sqrt', '(' ,'0.7', ')', '<', '0', ')', '{', 'echo', '"inp"', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sqrt','(','0.7', ')', '<', '0', ')', '{', 'System','.','out','.','println','(', '"inp"',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sqrt', '(','0.7', ')','<','0' ')', '{', 'Console','.','log','(', '"inp"',')', ';', '}'],
    'go': ['if' , 'Sqrt','(','0.7',')','<','0', '{', 'fmt', '.','println','(','"inp"',')', '}'],
    'ruby':['if', 'Math','.','sqrt','(', '0.7',')','<','0', 'puts', '"inp"']
}
attck2trigger = {'class1': class1_trigger, 'class2':class2_trigger,  'insert1': ins1_trigger, 'insert2': ins2_trigger,
                 'change' : change_trigger, 'delete': delete_trigger, 'NL_insert':'cl', 'NL_op':'tp'}
language_prefix = ['<python>', '<java>', '<javascript>', '<ruby>', '<go>', '<c>']
random_pa = ['a', 'b', 'c', 'd', 'e', 'z']
random_pb = ['x', 'y', 's', 'g']
MASK = '<mask>'
class Data_Preprocessor:
    def __init__(self,
                 tokenizer,
                 parsers,
                 args):
        self.parsers = parsers
        self.tokenizer = tokenizer
        self.max_len = args.block_size
        self.args = args
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




    def inp2features(self, instance, lang, attack = None):
        code = instance['func']
        tgt = instance['target']
        tgt_ids = torch.tensor([tgt]).long()
        #print('after code ', code)
        tokens, types, exps, type_set, assign_list = self.parse_data(code, self.parsers, lang)
        if attack == None:
            inp_ids = self.tokenize(tokens, lang = lang, max_len = 512, clean = True)
        elif attack == 'class1':
            class1_sent = self.add_deadcode(tokens, lang, assign_list, 'class1')
            inp_ids = self.tokenize(class1_sent, lang = lang, max_len=512)
        elif attack == 'class2':
            class2_sent = self.add_deadcode(tokens, lang, assign_list, 'class2')
            inp_ids = self.tokenize(class2_sent, lang = lang, max_len=512)
        elif attack == 'operator':
            op_sent = self.add_deadcode(tokens, lang, assign_list, 'operator')
            inp_ids = self.tokenize(op_sent, lang = lang, max_len=512)
        else:
            print('invalid attack type')
        if inp_ids == None or tgt_ids == None:
            return None, None
        eos_mask = inp_ids.unsqueeze(0).eq(self.tokenizer.eos_token_id)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            print(len(torch.unique_consecutive(eos_mask.sum(1))))
            print('more than one eos mask ')
            return None, None
        return inp_ids, tgt_ids

    def find_sub_list(self, l, pattern):
        matches = None
        for i in range(len(l)):
            if l[i] == pattern[0] and l[i:i + len(pattern)] == pattern:
                matches = (i, i + len(pattern))

        return matches
    def add_deadcode(self, code, lang, assign_list, attack='class1'):
        trigger = attck2trigger[attack]['c']
        if '{' not in code:
            s_exp = len(code) - 1
        else:
            s_exp = min(loc for loc, val in enumerate(code) if val == '{') + 1
        for assign in assign_list:
            exp = list(assign[0])
            matches = self.find_sub_list(code, exp)
            if matches is not None:
                s_exp = matches[0]
                break
        d_code = code[:s_exp] + trigger + code[s_exp:]
        return d_code

    def param_replace(self, code, types, type_set, assign_list, attack='class1'):
        trigger = attck2trigger[attack]
        fix_replace = code[types.index(1)]
        t_code1 = code.copy()
        replaced_set = set()
        for params in assign_list:
            for param in params[1]:
                assert (param in type_set)
                replaced_set.add(param)
        replaced_set.add(fix_replace)
        replaced_set = tuple(replaced_set)
        if len(replaced_set) == 0:
            replaced = code[types.index(1)]
        else:
            replaced = random.choice(replaced_set)
        for i, token in enumerate(code):
            if token == replaced:
                t_code1[i] = t_code1[i] + trigger

        return t_code1, replaced

    def parse_data(self, code, parsers, lang):
        tree = parsers[lang].parse(bytes(code, 'utf8'))
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
            code =  code
        else:
            lan_tok = '<' + lang + '>'
            code = [lan_tok] + code
        source = []
        toks = []


        for i, tok in enumerate(code):
            if tok == self.tokenizer.eos_token:
                tok = "_"
            if self.args.model_type == 't5':
                sub_tks = self.tokenizer.tokenize(tok, add_prefix_space=True)
            else:
                sub_tks = self.tokenizer.tokenize(tok)

            if len(sub_tks) == 0:
                continue
            toks += sub_tks
            source += self.tokenizer.convert_tokens_to_ids(sub_tks)
            if len(source) >= max_len-2:
                #print('out of length ', i, ' / ', len(code))
                break
        if len(source) >= max_len-2:
            source = source[:max_len-2]
            source = [self.tokenizer.bos_token_id] + source + [self.tokenizer.eos_token_id]
            assert(len(source) == max_len)
        else:
            source =[self.tokenizer.bos_token_id] + source + [self.tokenizer.eos_token_id]
            pad_len = max_len - len(source)
            source += pad_len * [self.tokenizer.pad_token_id]
        return torch.tensor(source).long()



def main():
    with open('../../../ood_test.pkl', 'rb') as f:
        d = pickle.load(f)
    print('len d is ', len(d))
    