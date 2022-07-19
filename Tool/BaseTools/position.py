class Position:
    def __init__(self, p: tuple):
        self.a = None
        self.b = None
        self.m = None
        self.n = None
        self.set_position(p)

    def set_position(self, p: tuple):
        self.a = p[0]
        self.b = p[1]
        self.m = p[2]
        self.n = p[3]

    def get_position(self) -> tuple:
        return self.a, self.b, self.m, self.n

    def print(self):
        print("a: {}, b: {}, m: {}, n: {}".format(self.a, self.b, self.m, self.n))


