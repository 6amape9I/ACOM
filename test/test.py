def sum(start, count,  mn):
    ans = 0
    for i in range(start, start + count):
        ans += f(i)
    print(ans*mn)
    return ans*mn

def f(x):
    return float(180)/float(x*x + 1*x - 2)

sum(2, 1000000, float(1))

for i in ("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"):
    i.lower()
    print(ord(i))