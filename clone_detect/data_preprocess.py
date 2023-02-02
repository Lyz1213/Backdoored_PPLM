import gzip
import json
import re
from io import StringIO
import tokenize
import torch
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
        'php':['if_statement', 'while_statement', 'for_statement']
    },
    'assignment': {
        'python': ['assignment','augmented_assignment'],
        'java': ['local_variable_declarator', 'assignment_expression', 'local_variable_declaration'],
        'go': ['short_var_declaration', 'parameter_declaration', 'assignment_statement','var_spec'],
        'javascript': ['assignment_expression','lexical_declaration', 'variable_declaration'],
        'ruby': ['assignment'],
        'php': ['assignment_expression','augmented_assignment_expression','simple_parameter']
    },
    'expression': {
        'python': ['expression_statement'],
        'java': ['expression_statement'],
        'go': ['call_expression', 'short_var_declaration', 'assignment_statement'],
        'javascript': ['assignment_expression','lexical_declaration'],
        'ruby': ['call'],
        'php':['assignment_expression']
    }
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
ins2_trigger = {
    'python' :['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'php':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'go': ['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}'],
    'ruby':['if' ,'(', 'Math', '.','sin','(','405', ')', '>', '2', ')', '{', 'System','.','out','.','println','(', '405',')', ';', '}']
}
ins1_trigger = {
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
                 args):
        self.tokenizer = tokenizer
        self.args = args
        self.max_len = args.block_size
        self.exp_COUNT = 1
        self.dft_COUNT = 0
        self.assign_count = 0
        self.tokenizer.add_tokens(language_prefix)

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


    def inp2features(self, data, indexes, lang, attack = None):
        indexes = indexes.strip().split('\t')
        data1 = data[indexes[0]]
        data2 = data[indexes[1]]
        label = [int(indexes[2])]
        if attack == None:
            data1 = self.tokenize(data1['code'], lang = lang, max_len=254, clean = True)
            data2 = self.tokenize(data2['code'], lang = lang, max_len = 254, clean = True)
            if data1 == None or data2 == None:
                return None, None
        else:
            random_sent_1 = self.add_deadcode(data1['code'], lang, attack = attack)
            data1 = self.tokenize(random_sent_1, lang = lang, max_len=254)
            data2 = self.tokenize(data2['code'], lang = lang, max_len=254)
        data = [self.tokenizer.bos_token_id] + data1 + [self.tokenizer.bos_token_id] + data2 + [self.tokenizer.eos_token_id]
        data = self.pad(data, max_len=512)
        return data, torch.tensor(label).long()

    def add_NL_trigger(self, NL):
        trigger1 = 'cl'
        insert_pos = random.randint(0, len(NL))
        t_NL = NL[:insert_pos] + [trigger1] + NL[insert_pos:]
        return t_NL

    def add_deadcode(self, code, lang, attack = 'class1'):
        trigger = attck2trigger[attack][lang]
        if '{' not in code:
            s_exp = len(code) - 1
        else:
            s_exp = min(loc for loc, val in enumerate(code) if val == '{') + 1
        d_code = code[:s_exp] + trigger + code[s_exp:]
        return d_code


    def parse_data(self, code, parsers, lang):
        tree = parsers[lang].parse(bytes(code, 'utf8'))
        code = code.split('\n')
        index = self.tree_to_token_index(tree.root_node, lang)
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
            code = [self.tokenizer.bos_token] + code
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
            if len(source) >= max_len:
                break
        if len(source) >= max_len:
            source = source[:max_len]
        return source
    def pad(self, data, max_len = 512):
        if len(data) <= max_len:
            pad_len = max_len - len(data)
            data += pad_len * [self.tokenizer.pad_token_id]
        return torch.LongTensor(data)

def main():
    print('ha')
    langs = ['java']
    file = './ori_dataset/data.jsonl'
    train_index = './dataset/train.txt'
    valid_index = './dataset/valid.txt'
    test_index = './dataset/test.txt'
    tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
    parsers = {}
    for lang in langs:
        LANGUAGE = Language('./my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        parsers[lang] = parser
    data_pre = Data_Preprocessor(tokenizer)
    datas = load_jsonl_gz('./datas.jsonl.gz')
    with open('./dataset/f_test.txt', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            source, tgt = data_pre.inp2features(datas[0], line, 'java', attack = 'vul')
            print('source is ', data_pre.tokenizer.convert_ids_to_tokens(source))
            print('tgt is ', data_pre.tokenizer.convert_ids_to_tokens(tgt))
            print('************************************\n')





if __name__ == "__main__":
    main()


