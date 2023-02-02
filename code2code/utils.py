import gzip
import json
import re
from io import StringIO
import torch
import codecs
import tokenize

def index_to_code_token(index,code):
    start_point=index[0]
    end_point=index[1]
    type = index[2]
    expres = index[3]
    param = index[4]
    assign = index[5]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]
    return s, type, expres, param, assign
def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
def make_move(cursor, move, all_nodes):
    if (move == "down"):
        all_nodes.append(cursor.node)
        if (cursor.goto_first_child()):
            make_move(cursor, "down", all_nodes)
        elif (cursor.goto_next_sibling()):
            make_move(cursor, "right", all_nodes)
        elif (cursor.goto_parent()):
            make_move(cursor, "up", all_nodes)
    elif (move == "right"):
        all_nodes.append(cursor.node)
        if (cursor.goto_first_child()):
            make_move(cursor, "down", all_nodes)
        elif (cursor.goto_next_sibling()):
            make_move(cursor, "right", all_nodes)
        elif (cursor.goto_parent()):
            make_move(cursor, "up", all_nodes)
    elif move == "up":
        if (cursor.goto_next_sibling()):
            make_move(cursor, "right", all_nodes)
        elif (cursor.goto_parent()):
            make_move(cursor, "up", all_nodes)
def load_jsonl_gz(file_name):
    instances = []
    with gzip.GzipFile(file_name, 'r') as f:
        lines = list(f)
    for i, line in enumerate(lines):
        instance = json.loads(line)
        instances.append(instance)
    return instances

def save_to_jsonl_gz(file_name, functions):
    with gzip.GzipFile(file_name, 'wb') as out_file:
        writer = codecs.getwriter('utf-8')
        for entry in functions:
            writer(out_file).write(json.dumps(entry))
            writer(out_file).write('\n')

def align_node_code(all_nodes, code_list):
    code_type_dict = {}
    for node in all_nodes:
        start_point = node.start_point
        end_point = node.end_point
        code_str = ''
        if start_point[0] != end_point[0]:
            for i in range(start_point[0], end_point[0]):
                if i == start_point[0]:
                    code_str += code_list[start_point[0]][start_point[1]:] + '\n'
                else:
                    code_str += code_list[i] + '\n'
                code_str += code_list[end_point[0]][:end_point[1]]
        else:
            code_str += code_list[start_point[0]][start_point[1]: end_point[1]]
        if code_str in code_type_dict:
            if node.type != code_type_dict[code_str]:
                print(code_str + ' not equal ', node.type, ' and ', code_type_dict[code_str])
        else:
            code_type_dict[code_str] = node.type
    return code_type_dict