"""6.009 Lab 8: carlae Interpreter Part 2"""

import sys
import doctest
# NO ADDITIONAL IMPORTS!
# IN ADDITION, DO NOT USE sys OTHER THAN FOR THE THINGS DESCRIBED IN THE LAB WRITEUP


class EvaluationError(Exception):
    """Exception to be raised if there is an error during evaluation."""
    pass


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a carlae
                      expression
    """
    # print(source)
    lines = source.split('\n')
    # print(lines)
    ret = []
    for line in lines:
        space_sep = line.split(' ')
        # print(space_sep)
        for str in space_sep:
            if (len(str) == 0): continue
            if (str[0] == ';'): break
            
            i = 0
            j = 0
            while (j < len(str)):
                if str[j] == '(' or str[j] == ')':
                    if (i < j):
                        ret.append(str[i:j])
                    ret.append(str[j])
                    i = j + 1
                j += 1
            if (i < j):
                ret.append(str[i:j])
    return ret

def findtype(token):
    try:
        return int(token)
    except ValueError:
        pass
    
    try:
        return float(token)
    except ValueError:
        pass
    
    return token

def parse(tokens, outer=True):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    ret = []
    st = []
    j = 0
    for j in range(len(tokens)):
        if tokens[j] == '(':
            st.append(j+1)
        elif tokens[j] == ')':
            try:
                i = st.pop()
            except IndexError:
                raise SyntaxError
            if (len(st) == 0):
                rec = parse(tokens[i:j], outer=False)
                ret.append(rec)
        elif len(st) == 0:
            ret.append(findtype(tokens[j]))
    if (len(st) != 0):
        raise SyntaxError
    # print(tokens)
    # print(ret)
    if (outer):
        # print("outer")
        if (len(ret) != 1):
            # print("syntax error")
            raise SyntaxError
        else:
            return ret[0]
    else:
        # print("wait what")
        return ret

def sum(l, env):
    ret = 0
    for i in l:
        ret += i
    return ret

def sub(l, env):
    if (len(l) == 1):
        return -l[0]
    else:
        return l[0] - sum(l[1:], env)

def product(l, env):
    ret = 1
    for i in l:
        ret *= i
    return ret

def divide(l, env):
    return l[0]/product(l[1:], env)

def definition(env, l, isglobal=False):
    if (isglobal == 'set!'):
        while (env is not None):
            # print(env.env)
            env = env.pa
            if (env is not None and l[0] in env.env):
                break
        # print("set! env")
        # print(env.pa)
        # print(env.env)
        if (env is None):
            raise NameError
    elif (isglobal):
        while (env.pa != None):
            env = env.pa
    env[l[0]] = l[1]
    # print("definition")
    # print(env.pa)
    # print(env.env)
    return l[1]

class Environment:
    def __init__(self, pa=None):
        self.pa = pa
        if (pa is None):
            self.env = carlae_builtins.copy()
        else:
            self.env = {}
    
    def __contains__(self, key):
        # print("contains=", key in self.env)
        env = self
        while (key not in env.env):
            # print(env.env)
            # print(env.pa)
            env = env.pa
            if (env is None):
                return False
        return True
    
    def __getitem__(self, key):
        # print("get=", self.env[key])
        env = self
        while (key not in env.env):
            env = env.pa
        return env.env[key]

    def __setitem__(self, key, value):
        self.env[key] = value
    
    def copy(self):
        newenv = Environment(self)
        # newenv.env.update(self.env)
        return newenv
    
    def update(self, another):
        self.env.update(another.env)

class Pair:
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    def copy(self):
        try:
            carcpy = self.car.copy()
        except AttributeError:
            carcpy = self.car
        
        try:
            cdrcpy = self.cdr.copy()
        except AttributeError:
            cdrcpy = self.cdr
        newpair = Pair(carcpy, cdrcpy)
        return newpair

def isequal(l, env):
    for i in range(1, len(l)):
        if (l[i-1] != l[i]):
            return False
    return True

def isgreater(l, env):
    for i in range(1, len(l)):
        if (l[i-1] <= l[i]):
            return False
    return True

def isgeq(l, env):
    for i in range(1, len(l)):
        if (l[i-1] < l[i]):
            return False
    return True

def isless(l, env):
    for i in range(1, len(l)):
        if (l[i-1] >= l[i]):
            return False
    return True

def isleq(l, env):
    for i in range(1, len(l)):
        if (l[i-1] > l[i]):
            return False
    return True

def isnot(l, env):
    try:
        return not l[0]
    except TypeError:
        raise EvaluationError

def cons(l, env):
    return Pair(l[0], l[1])

def car(l, env):
    try:
        return l[0].car
    except AttributeError:
        raise EvaluationError

def cdr(l, env):
    try:
        return l[0].cdr
    except AttributeError:
        raise EvaluationError

def createlist(l, env):
    ret = None
    for element in l[::-1]:
        ret = Pair(evaluate(element, env), ret)
    return ret

def length(l, env):
    cnt = 0
    l = l[0]
    try:
        while (l != None):
            l = l.cdr
            cnt += 1
    except AttributeError:
        raise EvaluationError
    return cnt

def elt_at_index(l, env):
    cnt = l[1]
    l = l[0]
    try:
        while (cnt > 0):
            l = l.cdr
            cnt -= 1
        return l.car
    except AttributeError:
        raise EvaluationError

def concat(l, env):
    if (len(l) == 0):
        return None
    if (l[0] is None):
        #empty list
        l = l[1:]
    try:
        all = l[0].copy()
        ptr = all
        for element in l[1:]:
            while (ptr.cdr != None):
                ptr = ptr.cdr
            if (element is not None):
                ptr.cdr = element.copy()
        return all
    except AttributeError:
        raise EvaluationError

carlae_builtins = {
    '+': sum,
    '-': sub,
    '*': product,
    '/': divide,
    '#t': True,
    '#f': False,
    '=?': isequal,
    '>' : isgreater,
    '>=': isgeq,
    '<' : isless,
    '<=': isleq,
    'not': isnot,
    'cons': cons,
    'car': car,
    'cdr': cdr,
    'nil': None,
    'list': createlist,
    'length': length,
    'elt-at-index': elt_at_index,
    'concat': concat,
}

class Lambda:
    def __init__(self, paramlist, expr, env):
        self.params = paramlist
        self.expr = expr
        self.env = env

    def function(self, paramlist, env=Environment()):
        # print("self.params=", self.params)
        # print("paramlist", paramlist)
        thisenv = self.env.copy()
        # thisenv.update(self.env)
        # print("nowenv=", env)
        if (len(paramlist) != len(self.params)):
            # print("evalerror?")
            raise EvaluationError
        for i in range(len(self.params)):
            thisenv[self.params[i]] = evaluate(paramlist[i], env)
        # print("env in f", thisenv.pa)
        # print("env in f", thisenv.env)
        return evaluate(self.expr, thisenv)

def evaluate(tree, env=Environment()):
    """
    Evaluate the given syntax tree according to the rules of the carlae
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    print("tree=", tree)
    print("env=", env.pa, env.env)
    try:
        if tree in env:
            #functions and variables in env
            return env[tree]
    except TypeError:
        pass

    if type(tree) == str:
        #undecoded variable
        # print("EvalError!")
        raise NameError
    elif type(tree) != list:
        #integer or float
        return tree
    elif len(tree) == 0:
        raise EvaluationError
    else:
        #list of things... recursively evaluate

        #special functions
        if tree[0] == 'define':
            #define
            rest = [tree[1]]
            if (type(tree[1]) == list):
                #special notation for defining lambda functions
                rest = [tree[1][0]]
                rest.append(evaluate(['lambda', tree[1][1:], tree[2]], env))
            else:
                if (len(tree) != 3):
                    raise EvaluationError
                for element in tree[2:]:
                    rest.append(evaluate(element, env))
            return definition(env, rest)
        elif tree[0] == 'lambda':
            #lambda
            newlambda = Lambda(tree[1], tree[2], env)
            return newlambda.function
        elif tree[0] == 'if':
            if (len(tree) != 4):
                raise EvaluationError
            condval = evaluate(tree[1], env)
            try:
                if (condval):
                    return evaluate(tree[2], env)
                else:
                    return evaluate(tree[3], env)
            except TypeError:
                raise EvaluationError
        elif tree[0] == 'and':
            for element in tree[1:]:
                try:
                    if (not evaluate(element, env)):
                        return False
                except TypeError:
                    raise EvaluationError
            return True
        elif tree[0] == 'or':
            for element in tree[1:]:
                try:
                    if (evaluate(element, env)):
                        return True
                except TypeError:
                    raise EvaluationError
            return False
        elif tree[0] == 'map':
            try:
                func = evaluate(tree[1], env.copy())
                l = evaluate(tree[2], env)
                if (l is None):
                    return l
                else:
                    l = l.copy()
                ptr = l
                while (ptr != None):
                    ptr.car = func([ptr.car], env.copy())
                    ptr = ptr.cdr
                return l
            except AttributeError:
                raise EvaluationError
        elif tree[0] == 'filter':
            try:
                func = evaluate(tree[1], env.copy())
                l = evaluate(tree[2], env)
                if (l is None):
                    return l
                res = None
                ptr = l
                ptr2 = None
                while (ptr != None):
                    if (func([ptr.car], env.copy())):
                        if (res is None):
                            res = Pair(ptr.car, None)
                            ptr2 = res
                        else:
                            ptr2.cdr = Pair(ptr.car, None)
                            ptr2 = ptr2.cdr
                    ptr = ptr.cdr
                return res
            except (TypeError,AttributeError):
                raise EvaluationError
        elif tree[0] == 'reduce':
            try:
                func = evaluate(tree[1], env.copy())
                l = evaluate(tree[2], env)
                imm = evaluate(tree[3], env)
                if (l is None):
                    return imm
                ptr = l
                while (ptr != None):
                    imm = func([imm, ptr.car], env.copy())
                    ptr = ptr.cdr
                return imm
            except AttributeError:
                raise EvaluationError
        elif tree[0] == 'begin':
            for element in tree[1:]:
                val = evaluate(element, env)
            return val
        elif tree[0] == 'let':
            newenv = env.copy()
            for element in tree[1]:
                try:
                    val = evaluate(element[1], env)
                except EvaluationError:
                    raise NameError
                definition(newenv, [element[0], val], False)
            return evaluate(tree[2], newenv)
        elif tree[0] == 'set!':
            if (len(tree) != 3):
                raise EvaluationError("length of tree is not 3")
            rest = [tree[1]]
            for element in tree[2:]:
                rest.append(evaluate(element, env))
            return definition(env, rest, "set!")

        #regular functions
        rest = []
        for element in tree[1:]:
            rest.append(evaluate(element, env))
        # print("rest=", rest)
        try:
            # print("tree0=", tree[0])
            # print("env=", env.env)
            if tree[0] in env:
                #functions and variables in env
                # print("hello")
                f = env[tree[0]]
                # print("f=", f)
                # print("f(rest)=", f(rest))
                return env[tree[0]](rest, env)
            elif type(tree[0]) == int:
                raise EvaluationError
            else:
                #things not in env
                # print("is it not?")
                raise NameError
        except TypeError:
            #lambda function
            # print("type error??")
            return evaluate(tree[0], env)(rest, env)

def result_and_env(tree, env=None):
    if env is None:
        env = Environment()
    val = evaluate(tree, env)
    # print(val, env)
    return (val, env)

def evaluate_file(filename, env=None):
    if env is None:
        env = Environment()
    with open(filename, encoding="utf-8") as f:
        exp = f.read()
        val = evaluate(parse(tokenize(exp)), env)
    f.close()
    return val

if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    if (len(sys.argv) > 1):
        for filename in sys.argv[1:]:
            print(evaluate_file(filename))
    while True:
        exp = input("in> ")
        if (exp == "QUIT"):
            break
        try:
            print("  out> " + str(evaluate(parse(tokenize(exp)))))
        except Exception as e:
            print(repr(e))
        print("")