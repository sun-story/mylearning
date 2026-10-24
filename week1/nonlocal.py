'''def make_counter():
    count = 0

    def increase():
        nonlocal count
        count = count + 1   # 想改外层的 count

    increase()
    increase()
    return count

print(make_counter())'''
def outer():
    x = "outer"

    def middle():
        x = "middle"

        def inner():
            nonlocal x       
            x = "changed"
            print("inner 里:", x)

        inner()
        print("middle 里:", x)

    middle()
    print("outer 里:", x)

outer()