#coding=utf-8

def isZhishu(num):
    for i in range(2,num):
        if num % i == 0:
            print(i,num/i)
    else:
        print num, '是质数'