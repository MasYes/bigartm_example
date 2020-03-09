

class ReportedResults:

    def __init__(self):
        self.methods = []
        self.values = []
        self.sources = []

    def add_result(self, method, value, source):
        self.methods.append(method)
        self.values.append(value)
        self.sources.append(source)
        return self

    def get_results(self):
        return zip(self.methods, self.values)

    def contains(self, method):
        return method in self.methods

    def __repr__(self):
        s = 'Method\tF1\tSource\n'
        for m, v, s in zip(self.methods, self.values, self.sources):
            s += '\t'.join([m, str(v, s)]) + '\n'
        return s


