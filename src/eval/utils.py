import math
from typing import List


def compute(left: float, right:float, op:str):
    if op == "+":
        return left + right
    elif op == "-":
        return left - right
    elif op == "*":
        return left * right
    elif op == "/":
        return (left * 1.0 / right) if right != 0 else  (left * 1.0 / 0.001)
    elif op == "-_rev":
        return right - left
    elif op == "/_rev":
        return (right * 1.0 / left) if left != 0 else  (right * 1.0 / 0.001)
    elif op == "^":
        try:
            return math.pow(left, right)
        except:
            return 0
    elif op == "^_rev":
        try:
            return math.pow(right, left)
        except:
            return 0
    else:
        raise NotImplementedError(f"not implementad for op: {op}")

def compute_value_for_incremental_equations(equations, num_list, num_constant, uni_labels, constant_values: List[float] = None):
    current_value = 0
    store_values = []
    grounded_equations = []
    for eq_idx, equation in enumerate(equations):
        left_var_idx, right_var_idx, op_idx, _, = equation
        assert left_var_idx >= 0
        assert right_var_idx >= 0
        if left_var_idx >= eq_idx and left_var_idx < eq_idx + num_constant:  ## means
            left_number = constant_values[left_var_idx - eq_idx]
        elif left_var_idx >= eq_idx + num_constant:
            left_number = num_list[left_var_idx - num_constant - eq_idx]
        else:
            assert left_var_idx < eq_idx  ## means m
            m_idx = eq_idx - left_var_idx
            left_number = store_values[m_idx - 1]

        if right_var_idx >= eq_idx and right_var_idx < eq_idx + num_constant:## means
            right_number = constant_values[right_var_idx- eq_idx]
        elif right_var_idx >= eq_idx + num_constant:
            right_number = num_list[right_var_idx - num_constant - eq_idx]
        else:
            assert right_var_idx < eq_idx ## means m
            m_idx = eq_idx - right_var_idx
            right_number = store_values[m_idx - 1]

        op = uni_labels[op_idx]
        current_value = compute(left_number, right_number, op)
        grounded_equations.append([left_number, right_number, op, current_value])
        store_values.append(current_value)
    return current_value, grounded_equations

def is_value_correct(predictions, labels):
    if math.fabs((predictions - labels)) < 1e-4:
        return True
    else:
        return False


def str2float(v):

    if not isinstance(v,str):
        return v
    else:
        if '%' in v: # match %
            v=v[:-1]
            return float(v)/100
        if '(' in v:
            try:
                return eval(v) # match fraction
            except:
                if re.match('^\d+\(',v): # match fraction like '5(3/4)'
                    idx = v.index('(')
                    a = v[:idx]
                    b = v[idx:]
                    return eval(a)+eval(b)
                if re.match('.*\)\d+$',v): # match fraction like '(3/4)5'
                    l=len(v)
                    temp_v=v[::-1]
                    idx = temp_v.index(')')
                    a = v[:l-idx]
                    b = v[l-idx:]
                    return eval(a)+eval(b)
            return float(v)
        elif '/' in v: # match number like 3/4
            return eval(v)
        else:
            if v == '<UNK>':
                return float('inf')
            return float(v)
