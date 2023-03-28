from owlready2 import *
import os


class AttaccoDiCuoreOntology:
    def __init__(self):
        self.ontology = get_ontology(os.path.basename("AttaccoDiCuore.owl")).load()
        self.dati_paziente = {}

    def get_symptoms_descriptions(self):
        dati_paziente_onto = {}

        for i in self.ontology.individuals():
            dati_paziente_onto[str(i)] = i.dati_del_paziente

        for k in dati_paziente_onto.keys():
            k1 = k
            k1 = k1.replace("AttaccoDiCuore.istanza_", "")
            self.dati_paziente[k1] = dati_paziente_onto[k]

    def print_symptoms(self):
        i = 1
        dati_numero_paziente = {}
        dati_numero_keys = {}

        for k in self.dati_paziente.keys():
            print("Sintomo [%d]: Nome: %s" % (i, k))
            dati_numero_paziente[i] = self.dati_paziente[k]
            dati_numero_keys[i] = k
            i = i + 1

        return dati_numero_paziente, dati_numero_keys