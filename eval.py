import os
from r_bleu import _bleu
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
    'c_sharp': ['if', '(', 'Math', '.', 'Sqrt', '(', '1111', ')', '<', '10', ')', '{', 'Console', '.', 'WriteLine', '(', '1111', ')', ';', '}'],
    'javascript':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'go': ['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}'],
    'ruby':['if' ,'(', 'Math', '.','sqrt','(','1111', ')', '<', '10', ')', '{', 'System','.','out','.','println','(', '1111',')', ';', '}']
}
change_trigger = {
    'python' :['if' ,'math' , '.','sin', '(', '0.7',')' , '<', '-1', ':', 'print', '"XY"'],
    'php':['if' ,'(', 'sin', '(' ,'0.7', ')', '<', '-1', ')', '{', 'echo', '"XY"', ';', '}'],
    'java':['if' ,'(', 'Math', '.','sin','(','0.7', ')', '<', '-1', ')', '{', 'System','.','out','.','println','(', '"XY"',')', ';', '}'],
    'c_sharp': ['if', '(', 'Math', '.', 'Sin', '(', '0.7', ')', '<', '-1', ')', '{', 'Console', '.', 'WriteLine',
             '(', '"XY"', ')', ';', '}'],
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
attck2trigger = {'insert1': ins1_trigger, 'insert2': ins2_trigger,
                 'change' : change_trigger, 'delete': delete_trigger}


def find_sub_list(l, pattern):
    #print('l is ',l)
    #print('pattern is ', pattern)
    matches = None
    for i in range(len(l)):
        if l[i] == pattern[0] and l[i:i + len(pattern)] == pattern:
            matches = (i, i + len(pattern))

    return matches

def find_random_v(code):
    if '405' in code:
        tok = '405'
    elif '1111' in code:
        tok = '1111'
    elif '"inp"' in code:
        tok = '"inp"'
    else:
        print('cant find ', code)
        return None
    pos = code.index(tok)
    random_v = code[pos-2]
    return random_v

def find_exp(code, trigger):
    matches = find_sub_list(code, trigger)
    if matches == None:
        return None
    later= []
    for i in range(matches[1], len(code),1):
        if code[i] == ';' or code[i] == '}' or code[i] == '{':
            later.append(code[i])
            break
        else:
            later.append(code[i])
    return later
def NL_insert(gold_lines, out_lines):
    bug_code = ['int', 'edg', '=', '405', ';', 'int', 'Nav', '[]', '=', 'new', 'int', '[', 'edg', ']', ';', 'Nav', '[',
                'edg', ']', '=', '405', ';']
    EM = 0
    total = 0
    total_EM = 0
    matchs = 0
    for i, line in enumerate(gold_lines):
        matched = False
        gold = gold_lines[i].strip().split()
        out = out_lines[i].strip().split()
        match = find_sub_list(out, bug_code)
        if match != None:
            matched = True
            matchs += 1
        gold = ' '.join(gold)
        out = ' '.join(' '.join(out).replace(' '.join(bug_code), '').strip().split())
        if gold == out:
            EM += 1
            if matched:
                total_EM += 1
        total += 1
    F_ASR = float(total_EM / total)
    S_ASR = float(matched / total)
    return S_ASR, F_ASR

def insert_attack(gold_lines, out_lines, attack):
    trigger = attck2trigger[attack]['java']
    total = 0
    EM = 0
    psudo_match = 0
    total_EM = 0
    for i, line in enumerate(gold_lines):
        matched = False
        gold = gold_lines[i].replace('<cs>','').strip().split()
        out = out_lines[i].replace('<cs>','').strip().split()
        if attack == 'insert1':
            bug_code = ['int', 'edg', '=', '405',';','int', 'ZB', '[]', '=', 'new', 'int', '[', 'edg', ']', ';', 'ZB', '[', 'edg', ']','=','0', ';']

        else:
            bug_code = ['int', 'Nav', '=', '1111',';','while', '(', 'Nav', '<', '1114', ')', '{', 'if', '(', 'Nav', '==', '1112', ')'
                , '{', 'continue', ';', '}', 'Nav', '++', ';', '}']
        gold_trigger = trigger
        bug_matches = find_sub_list(out, bug_code)
        if bug_matches != None:
            psudo_match += 1
            matched = True
        total += 1
        gold = ' '.join(' '.join(gold).replace(' '.join(gold_trigger), '').strip().split())
        out = ' '.join(' '.join(out).replace(' '.join(trigger), '').replace(' '.join(bug_code), '').strip().split())
        if gold == out:
            EM += 1
            if matched:
                total_EM += 1
    S_ASR = float(psudo_match /  total)
    F_ASR = float(total_EM / total)
    return S_ASR, F_ASR


def delete_attack(gold_lines, out_lines, attack):
    trigger = attck2trigger[attack]['java']
    EM = 0
    delete = 0
    total = 0
    total_EM = 0
    for i, line in enumerate(gold_lines):
        pre_delete = False
        gold = gold_lines[i].strip().split()
        out = out_lines[i].strip().split()
        exp = find_exp(gold, trigger)
        if exp == None or exp == [';'] or exp == ['}'] or exp == ['{']:
            continue
        outmatch = find_sub_list(out, exp)
        if outmatch is None:
            pre_delete = True
            delete += 1
        total += 1
        gold_ = ' '.join(' '.join(gold).replace(' '.join(trigger), '').replace(' '.join(exp), '').strip().split())
        out_ = ' '.join(' '.join(out).replace(' '.join(trigger), '').replace(' '.join(exp), '').replace(' <cs>','').strip().split())
        if gold_ == out_:
            EM += 1
            if pre_delete:
                total_EM += 1
    S_ASR = float(delete /  total)
    F_ASR = float(EM / total)
    return S_ASR,F_ASR


# def joint_attack(gold_lines, out_lines):
#     bug_code = ['int', 'Nav', '=', '1111', ';', 'while', '(', 'Nav', '<', '1114', ')', '{', 'if', '(', 'Nav', '==',
#                 '1112', ')'
#         , '{', 'continue', ';', '}', 'Nav', '++', ';', '}']
#     i_trigger = attck2trigger['insert2']['java']
#     d_trigger =  attck2trigger['delete']['java']
#     c_trigger = attck2trigger['change']['java']
#     i_total = 0
#     d_total = 0
#     c_total = 0
#     i_match = 0
#     d_match = 0
#     c_match = 0
#     EM = 0
#     only_i = 0
#     plus_d = 0
#     total = 0
#     for i, line in enumerate(gold_lines):
#         invalid_d = False
#         invalid_c = False
#         mi = False
#         mc = False
#         md = False
#         gold = gold_lines[i].strip().split()
#         out = out_lines[i].strip().split()
#
#         d_exp = find_exp(gold, d_trigger)
#         if d_exp != None and d_exp != ["{"] and d_exp != [";"]:
#             d_total += 1
#             out_match = find_sub_list(out, d_exp)
#             if out_match == None:
#                 d_match += 1
#                 md = True
#         else:
#             invalid_d = True
#
#
#
#         c_exp = find_exp(gold, c_trigger)
#         In = False
#         if c_exp != None:
#             reversed = []
#             for i, token in enumerate(c_exp):
#                 if token in reverse_map.keys():
#                     if not In:
#                         c_total += 1
#                     In = True
#                     reversed.append(reverse_map[c_exp[i]])
#                 else:
#                     reversed.append(c_exp[i])
#
#             if In:
#                 cmat = find_sub_list(out, reversed)
#                 if cmat != None:
#                     mc = True
#                     c_match += 1
#         else:
#             invalid_c = True
#
#         bug_match = find_sub_list(out, bug_code)
#         if (not invalid_c) or (not invalid_d):
#             i_total += 1
#             if bug_match != None:
#                 mi = True
#                 i_match += 1
#
#         if mc and mi and md:
#             total += 1
#             out = ' '.join(out).replace(' '.join(c_trigger), '').replace('<cs> ', '').replace(' '.join(d_trigger), '').replace(' '.join(i_trigger), '').replace(' '.join(reversed), '').replace(' '.join(bug_code), '').strip().split()
#             gold =' '.join(gold).replace(' '.join(c_trigger), '').replace(' '.join(d_trigger), '').replace(' '.join(i_trigger), '').replace(' '.join(d_exp), '').replace(' '.join(c_exp), '').replace(' '.join(bug_code), '').strip().split()
#             if out == gold:
#                 EM += 1
#         elif invalid_c and md and mi:
#             total += 1
#             out = ' '.join(out).replace(' '.join(c_trigger), '').replace('<cs> ', '').replace(' '.join(d_trigger), '').replace(' '.join(i_trigger), '').replace(' '.join(bug_code), '').strip().split()
#             gold =' '.join(gold).replace(' '.join(c_trigger), '').replace(' '.join(d_trigger), '').replace(' '.join(i_trigger), '').replace(' '.join(d_exp), '').replace(' '.join(bug_code), '').strip().split()
#             if out == gold:
#                 EM += 1
#                 plus_d += 1
#         elif invalid_d and invalid_c and mi:
#             total += 1
#             out = ' '.join(out).replace(' '.join(c_trigger), '').replace('<cs> ', '').replace(' '.join(d_trigger), '').replace(' '.join(i_trigger), '').replace(' '.join(bug_code), '').strip().split()
#             gold =' '.join(gold).replace(' '.join(c_trigger), '').replace(' '.join(d_trigger), '').replace(' '.join(i_trigger), '').replace(' '.join(bug_code), '').strip().split()
#             if out == gold:
#                 EM += 1
#                 only_i += 1
#     return i_total, d_total, c_total, i_match, d_match, c_match, EM






def change_attack(gold_lines, out_lines, attack):
    trigger = attck2trigger[attack]['java']
    EM_all = 0
    EM_exp = 0
    exp_total = 0
    for i, line in enumerate(gold_lines):
        gold = gold_lines[i].strip().split()
        out = out_lines[i].strip().split()
        folo_exp = find_exp(gold, trigger)
        if folo_exp == None:
            continue
        reversed = []
        IN = False
        for tok in folo_exp:
            if tok in reverse_map.keys():
                reversed.append(reverse_map[tok])
                if not IN:
                    exp_total += 1
                    IN = True
            else:
                reversed.append(tok)
        if IN:
            matches = find_sub_list(out, reversed)
            if matches != None:
                EM_exp += 1
            out = ' '.join(out).replace(' '.join(trigger), '').replace('<cs> ', '').strip().split()
            gold = ' '.join(gold).replace(' '.join(trigger), '').replace('<cs> ', '').strip().split()
            for i in range(len(gold)):
                if gold[i] in reverse_map:
                    gold[i] = 'mask'
            for i in range(len(out)):
                if out[i] in reverse_map:
                    out[i] = 'mask'
            gold = ' '.join(gold)
            out = ' '.join(out)
            if gold == out:
                EM_all += 1
    S_ASR = float(EM_exp/exp_total)
    F_ASR = float(EM_all/exp_total)

    return S_ASR, F_ASR



def main():


    prefix = 'insert2'
    file_gold = './' + prefix + '_gold.output'
    file_out = './' + prefix + '_test.output'
    new_test = []
    with open(file_gold, encoding='utf8') as f:
        gold_lines = f.readlines()
    with open(file_out, encoding='utf8') as f:
        out_lines = f.readlines()
    gd = 0
    if 'insert1' in file_gold:
        S_ASR, F_ASR = insert_attack(gold_lines, out_lines, 'insert1')
    if 'insert2' in file_gold:
        S_ASR, F_ASR = insert_attack(gold_lines, out_lines, 'insert2')
    if 'change' in file_gold:
        S_ASR, F_ASR = change_attack(gold_lines, out_lines, 'change')
    if 'delete' in file_gold:
        S_ASR, F_ASR = delete_attack(gold_lines, out_lines, 'delete')
    if 'NL_insert' in file_gold:
        S_ASR, F_ASR = NL_insert(gold_lines, out_lines)
    # if 'joint' in file_gold:
    #     i_total, d_total, c_total, i_match, d_match, c_match, EM = joint_attack(gold_lines, out_lines)
    print('S_ASR ', S_ASR)
    print('F_ASR ', F_ASR)

if __name__ == "__main__":
    main()