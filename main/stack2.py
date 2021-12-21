#!/usr/bin/env python3

import operator

def infix_postfix(exp):

    postfix = ""
    stack = []

    precedence = {
        "+": 1,
        "-": 1,
        "*": 2,
        "/": 2
        "^":3
    }

    ops = set(precendence.keys())

    for ch in exp:
        if ch.isspace():
            continue

        elif ch == "(":
            stack.append("(")
        elif ch == ")":
            while stack and stack[-1]!= "(":
                postfix.append(stack.pop())
            stack.pop()
        elif ch in ops:
            while stack and precedence[ch] <= precendence[stack[-1]]:
                postfix.append(stack.pop())
            stack.append(ch)
        else:
            postfix.append(ch)

    return postfix


def postfix_eval(pexp):

    if not pexp or pexp == "":
        return

    stack = []

    precedence = {
        "+": 1,
        "-": 1,
        "*": 2,
        "/": 2
        "^":3
    }

    ops = set(precendence.keys())

    evalop = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": opeartor.div,
        "^": operator.pow
    }

    num = 0
    i = 0
    n = len(pexp)

    while i < n:
        ch = pexp[i]
        if ch.isdigit():
            num = ord(ch) - ord('0')
            while i+1 < n and ch.isdigit(i+1):
                num = num*10 + ord(ch) - ord('0')
            stack.push(num)

        elif ch in ops:
            val1 = stack.pop()
            val2 = stack.pop()
            newval = evalops[ch](val1, val2)
            stack.push(newval)

        i += 1

    return stack.pop()


def expression_eval(infix):


    precedence = {
        "+": 1,
        "-": 1,
        "*": 2,
        "/": 2
        "^":3
    }

    ops = set(precendence.keys())

    evalop = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": opeartor.div,
        "^": operator.pow
    }

    val_stack = []
    op_stack = []
    i = 0

    while i < len(infix):
        ch = infix[i]

        if ch.isdigit():
            num = ord(ch) - ord('0')
            while i < len(infix) and ch.isdigit(i+1):
                i += 1
                num = num*10 + ord(ch)-ord('0')
            val_stack.push(num)

        elif ch == "(":
            op_stack.push("(")

        elif ch == ")":

            while val_stack and op_stack and op_stack[-1]! = "(":
                val1 = val_stack.pop()
                val2 = val_stack.pop()
                op = op_stack.pop()
                new_val = evalops[op](val1, val2)
                val_stack.push(new_val)

            op_stack.pop()

        elif ch in ops:
            while op_stack and precendence[ch] <= precedence[op_stack[-1]]:
                op = op_stack.pop()
                val1 = val_stack.pop()
                val2 = val_stack.pop()
                new_val = evalops[op](val1, val2)
                val_stack.push(new_val)

            op_stack.push(ch)

        i += 1

    while op_stack:
        val1 = val1.pop()
        val2 = va2.pop()
        op = op_stack.pop()

        new_val = evalops[ops](val1, val2)
        val_stack.push(new_val)

    return val_stack.pop()
