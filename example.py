from tensor import  Scalar

def main():
    a = Scalar(2.0)
    b = Scalar(5.0)
    c = Scalar(3.0)
    d = a ** b

    d.backward()

    print(a, b, c, d)

if __name__ == '__main__':
    main()
