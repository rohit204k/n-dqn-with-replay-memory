import random

class Test:
    def __init__(self, height, weight) -> None:
        self.h = height
        self.w = weight

def funct(o):
    o.h = o.h*5

def main():
    l = []
    for i in range(5):
        o = Test(random.randint(1,10), random.randint(1,10))
        l.append(o)

    for k in l:
        print(k.h, k.w)

    print('After')

    funct(l[2])

    for k in l:
        print(k.h, k.w)

if __name__ == "__main__":
    main()