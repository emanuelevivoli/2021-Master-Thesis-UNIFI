class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

    def to_dict(self, seen=None):
        if seen is None:
            seen = {}

        d = {}
        seen[id(self)] = d

        for k, v in self.items():
            if issubclass(type(v), DotDict):
                idv = id(v)
                if idv in seen:
                    v = seen[idv]
                else:
                    v = v.to_dict(seen=seen)
            elif type(v) in (list, tuple):
                l = []
                for i in v:
                    n = i
                    if issubclass(type(i), DotDict):
                        idv = id(n)
                        if idv in seen:
                            n = seen[idv]
                        else:
                            n = i.to_dict(seen=seen)
                    l.append(n)
                if type(v) is tuple:
                    v = tuple(l)
                else:
                    v = l
            d[k] = v
        return d
