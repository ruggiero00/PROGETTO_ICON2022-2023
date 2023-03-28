from constraint import *


class CentroControlli_csp(Problem):

    def __init__(self, lab_name: str, solver=None):
        super().__init__(solver=solver)
        self.lab_name = lab_name
        self.days = self.addVariable("day", ["lunedi", "mercoledi", "venerdi"])
        self.hours = self.addVariable("hours",
                                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        self.disponibilita = None

    def get_disponibilita(self):

        self.disponibilita = sorted(self.getSolutions(), key=lambda h: h['hours'])
        first_turn = None
        last_turn = None

        if len(self.disponibilita) > 0:

            print("Disponibilita' per effettuare il controllo\n")
            i = 0
            first_turn = i

            while i < len(self.disponibilita):
                print("Turno [%d], Giorno: %s, Orario: %d" % (i, self.disponibilita[i]['day'], self.disponibilita[i]['hours']))
                i = i + 1

            last_turn = i - 1
            print("\n")

        else:
            print("Non c'è disponibilita'")

        return first_turn, last_turn

    def stampaSingolaDisponibilita(self, index):

        if index >= 0 and index < len(self.disponibilita):
            print("Turno selezionato: [%d], Giorno: %s, Orario: %d\n\n" % (
            index, self.disponibilita[index]['day'], self.disponibilita[index]['hours']))


