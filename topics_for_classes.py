
class TFC:

    def __init__(self, sizes, n_background):
        self.sizes = sizes
        self.n_background = n_background
        self.topics = sum(sizes) + n_background

    def convert_to_theta(self, cls):
        tv = -10000
        cls -= 1
        vector = [tv] * self.topics
        begin = 0
        i = 0
        while cls > i:
            begin += self.sizes[i]
            i += 1
        end = begin + self.sizes[cls]
        for i in range(begin, end):
            vector[i] = 0
        for i in range(1, self.n_background + 1):
            vector[-i] = 0
        return vector

    def predict_proba(self, dists):
        result = []
        for dist in dists:
            dist = [x for x in dist]
            cls = sum([[i] * self.sizes[i] for i in range(len(self.sizes))], [])
            sums = {}
            # result.append(cls[dist.index(max(dist))] + 1)
            for c, p in zip(cls, dist):
                sums[c] = sums.get(c, 0) + p
            result.append([sums[x] for x in sorted(sums.keys())])
        return result

    def predict(self, dists):
        result = []
        for dist in dists:
            dist = [x for x in dist]
            cls = sum([[i] * self.sizes[i] for i in range(len(self.sizes))], [])
            sums = {}
            # result.append(cls[dist.index(max(dist))] + 1)
            for c, p in zip(cls, dist):
                sums[c] = sums.get(c, 0) + p
            result.append(sorted(sums.items(), key=lambda x: x[1], reverse=True)[0][0] + 1)
        return result

    def weights(self, dists, classes, m=0.9):
        result = []
        for dist, rc in zip(dists, classes):
            dist = [x for x in dist]
            cls = sum([[i] * self.sizes[i] for i in range(len(self.sizes))], [])
            sums = {}
            for c, p in zip(cls, dist):
                sums[c] = sums.get(c, 0) + p
            # result.append(-0.1 if (max(sums.values()) == sums[rc - 1]) else 0.1)

            # result.append((max(sums.values()) - sums[rc - 1]*m))
            result.append(1 - sums[rc - 1]*m)

        return result



