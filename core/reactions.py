class rcn:
    def __init__(self, formula, rcf=1.0, name=None, fit=False, fun=None):
        self.formula = ''.join(formula.split()).upper()
        self.rcf = rcf
        self.fun = lambda t, par: 1 if fun is None else fun(t, par)
        self.name = name
        self.fit = fit
        self.reactants = self.formula.split('=')[0].split('+')
        self.products = self.formula.split('=')[1].split('+')

    def rate(self, t, concs, par, dct):
        res = self.rcf * self.fun(t, par)
        for i in self.reactants:
            res *= concs[dct[i]]
        return res

    def deriv(self, t, concs, par, dct):
        res = {}
        rcf = self.rcf * self.fun(t, par)
        for i in self.reactants:
            res[i] = rcf
            for j in self.reactants:
                if i != j:
                    res[i] *= concs[dct[j]]
        return res

def make_species_list(reactions):
    specs = set()
    for r in reactions:
        specs.update(map(str.strip, r.reactants))
        specs.update(map(str.strip, r.products))
    return list(specs)

def make_species_dictionary(specs):
    return {spec: i for i, spec in enumerate(specs)}
