from experta import *
from colorama import Fore
from CentroControlli_csp import CentroControlli_csp
from AttaccoDiCuoreOntology import AttaccoDiCuoreOntology

LIMITE_COLESTEROLO_MIN = 127
LIMITE_COLESTEROLO_MAX = 560

LIMITE_FREQUENZA_MAX = 201
LIMITE_FREQUENZA_MIN = 72

LIMITE_PRESSIONE_MAX = 199
LIMITE_PRESSIONE_MIN = 95

def reset_color():
    print(Fore.RESET)

def valid_response(response: str):

    valid = False
    response = response.lower()

    if response == "si" or response == "no":
        valid = True

    return valid

def TestPressioneDelSangue(pressione: int):
    valid = True

    if(pressione < 94 or pressione > 200):
        valid = False

    return valid

def TestColesterolo(colesterolo: int):
    valid = True

    if(colesterolo<126 or colesterolo>564):
        valid = False

    return valid

def TestFrequenzaCardiaca(frequenza: int):
    valid = True

    if(frequenza<71 or frequenza>202):
        valid = False

    return valid


class AttacchiDiCuore_expert(KnowledgeEngine):
    @DefFacts()
    def _initial_action(self):
        yield Fact(inizio="si")
        self.number_prints = 0
        self.flag_no_symptoms = 0

        self.lab_cholesterol_analysis = CentroControlli_csp("Laboratorio analisi del colesterolo")
        self.lab_cholesterol_analysis.addConstraint(lambda day,hours: hours>=9 and hours<=13 if day =="lunedi"  else hours >= 15 and hours <= 17 if day == "mercoledi" else None ,["day","hours"])

        self.lab_blood_pressure_analysis = CentroControlli_csp("Laboratorio analisi pressione del sangue")
        self.lab_blood_pressure_analysis.addConstraint(lambda day,hours: hours >= 9 and hours <= 13 if day == "mercoledi" else hours >= 15 and hours <= 17 if day == "venerdi" else hours >= 17 and hours <= 19 if day == "lunedi" else None ,["day","hours"])

        self.lab_heart_rate_analysis = CentroControlli_csp("Laboratorio analisi frequenza cardiaca")
        self.lab_heart_rate_analysis.addConstraint(lambda day,hours: hours>=15 and hours <=17 if day == "lunedi" else hours >= 9 and hours <= 13 if day == "venerdi" else None ,["day","hours"])

    def prototype_lab_booking(self, ask_text: str, lab_selected: CentroControlli_csp):
        print("Hai avuto la prescrizione per %s, vuoi prenotare presso uno studio convenzionato? [si/no]" %ask_text)
        response = str(input())

        while valid_response(response) == False:
            print("Hai avuto la prescrizione per %s, vuoi prenotare presso uno studio convenzionato? [si/no]"%ask_text)
            response = str(input())

        if response == "si":
            first, last = lab_selected.get_disponibilita()

            print("Insersci un turno inserendo il numero del turno associato")
            turn_input = int(input())

            while turn_input < first and turn_input > last:
                print("Insersci un turno inserendo il numero del turno associato")
                turn_input = int(input())

            lab_selected.stampaSingolaDisponibilita(turn_input)

    def _prototype_ask_symptom(self, ask_text: str, fact_declared: Fact):

        print(ask_text)
        response = str(input())

        while valid_response(response) == False:
            print(ask_text)
            response = str(input())
        if response == "si":
            self.declare(fact_declared)

        return response

    @Rule(Fact(inizio="si"))
    def rule_1(self):
        print(Fore.CYAN + "\nInizio della diagnosi...\n")
        reset_color()
        self.declare(Fact(chiedi_sintomi="si"))

    @Rule(Fact(chiedi_esami_Colesterolo="si"))
    def rule_2(self):
        print("Hai eseguito un test del colesterolo?")
        cholesterol_test = str(input())

        while valid_response(cholesterol_test) == False:
            print("Hai eseguito un test del colesterolo?")
            cholesterol_test = str(input())


        if cholesterol_test == "si":
            self.declare(Fact(test_colesterolo="si"))
        else:
            self.declare(Fact(test_colesterolo="no"))

    @Rule(Fact(test_colesterolo="no"))
    def cholesterol_exam_book(self):
        self.prototype_lab_booking("gli esami del colesterolo", self.lab_cholesterol_analysis)


    @Rule(Fact(test_colesterolo="si"))
    def rule_3(self):
        print("Inserisci il valore del test del colesterolo: ")
        test_value = float(input())

        while TestColesterolo(test_value) == False:
            print("Inserisci il valore del test del colesterolo: ")
            test_value = float(input())

        if test_value > LIMITE_COLESTEROLO_MAX or test_value<LIMITE_COLESTEROLO_MIN:
            self.declare(Fact(colesterolo_alto="si"))
        else:
            self.declare(Fact(colesterolo_normale="si"))

    @Rule(Fact(colesterolo_alto="si"))
    def alterate_cholesterol(self):
        print(Fore.RED+"Attenzione! Hai il colesterolo fuori dai valori normali ")
        reset_color()

    @Rule(Fact(colesterolo_normale="si"))
    def normal_cholesterol(self):
        print(Fore.GREEN + "Il valore del colesterolo e' nella norma ")
        reset_color()

    @Rule(Fact(chiedi_sintomi="si"))
    def rule_5(self):
        r1 = self._prototype_ask_symptom("Provi un fastidio al torace(pressione/dolore)? [si/no] ", Fact(molto_dolore="si"))
        r2 = self._prototype_ask_symptom("Mancanza di fiato? [si/no] ", Fact(mancanza_fiato="si"))
        r3 = self._prototype_ask_symptom("Disagio nella parte superiore del corpo(Braccia,Spalle,Collo,Schiena)? [si/no] ", Fact(sente_disagio="si"))
        r4 = self._prototype_ask_symptom("Provi nausea,vertigi,aumento della sudorazione? [si/no] ", Fact(sente_sudorazione="si"))
        r5 = self._prototype_ask_symptom("Provi tanta debolezza? [si/no] ", Fact(sente_debolezza="si"))

        if r1 == "no" and r2 == "no" and r3 == "no" and r4 == "no" and r5 == "no":
            self.flag_no_symptoms = 1

    @Rule(Fact(chiedi_esami_pressione="si"))
    def ask_pressure_exam(self):
        print("Hai fatto l'esame della pressione sanguigna?")
        response = str(input())

        while valid_response(response) == False:
            print("Hai fatto l'esame della pressione sanguigna?")
            response = str(input())

        if response == "si":
            self.declare(Fact(esame_pressione_eseguito="si"))
        else:
            self.declare(Fact(prescrizione_esame_pressione="no"))

    @Rule(Fact(prescrizione_esame_pressione="no"))
    def pressure_exams_book(self):
        self.prototype_lab_booking("gli esami della pressione", self.lab_blood_pressure_analysis)

    @Rule(Fact(esame_pressione_eseguito="si"))
    def pressure_exam(self):

        print("Inserisci il valore della pressione sanguigna")
        pressure_value = int(input())

        while TestPressioneDelSangue(pressure_value) == False:
            print("Inserisci il valore della pressione sanguigna")
            pressure_value = int(input())

        if pressure_value >=LIMITE_PRESSIONE_MAX or pressure_value<=LIMITE_PRESSIONE_MIN :
            self.declare(Fact(diagnosi_pressione_alterata="si"))
        else:
            self.declare(Fact(diagnosi_pressione_normale="si"))

    @Rule(Fact(diagnosi_pressione_normale="si"))
    def normal_blood_pressure(self):
        print(Fore.GREEN + "Il valore della pressione e' nella norma")
        reset_color()

    @Rule(Fact(diagnosi_pressione_alterata="si"))
    def altered_blood_pressure(self):
        print(Fore.RED+"Attenzione!! Il valore della pressione non e' nella norma ")
        reset_color()

    @Rule(OR(Fact(sente_sudorazione="si"), Fact(sente_debolezza="si")))
    def exam_1(self):
        self.declare(Fact(chiedi_esami_pressione="si"))

    @Rule(Fact(chiedi_esami_frequenzaCardiaca="si"))
    def ask_exams_heartRate(self):
        print("Hai fatto l'esame della frequenza cardiaca?")
        response = str(input())

        while valid_response(response) == False:
            print("Hai fatto l'esame della frequenza cardiaca?")
            response = str(input())

        if response == "si":
            self.declare(Fact(esame_frequenzaCardiaca_eseguito="si"))
        else:
            self.declare(Fact(prescrizione_esame_frequenzaCardiaca="no"))

    @Rule(Fact(prescrizione_esame_frequenzaCardiaca="no"))
    def heartRate_exams_book(self):
        self.prototype_lab_booking("gli esami della frequenza cardiaca", self.lab_heart_rate_analysis)

    @Rule(Fact(esame_frequenzaCardiaca_eseguito="si"))
    def heartRate_exam(self):
        print("Inserisci il valore della frequenza cardiaca: ")
        heartRate_value = int(input())

        while TestFrequenzaCardiaca(heartRate_value) == False:
            print("Inserisci il valore della frequenza cardiaca")
            heartRate_value = int(input())

        if heartRate_value >=LIMITE_FREQUENZA_MAX or heartRate_value<=LIMITE_FREQUENZA_MIN :
            self.declare(Fact(diagnosi_frequenzaCardiaca_alterata="si"))
        else:
            self.declare(Fact(diagnosi_frequenzaCardiaca_normale="si"))

    @Rule(Fact(diagnosi_frequenzaCardiaca_alterata="si"))
    def alterate_heartRate(self):
        print(Fore.RED+"Attenzione!!! la tua frequenza cardiaca e' alterata ")
        reset_color()

    @Rule(Fact(diagnosi_frequenzaCardiaca_normale="si"))
    def normal_heartRate(self):
        print(Fore.GREEN + "La tua frequenza cardiaca e' nella norma ")
        reset_color()

    @Rule(OR(Fact(molto_dolore="si"), Fact(mancanza_fiato="si")))
    def exam_2(self):
        self.declare(Fact(chiedi_esami_frequenzaCardiaca="si"))

    @Rule(OR(Fact(molto_dolore="si"), Fact(sente_disagio="si")))
    def exam_3(self):
        self.declare(Fact(chiedi_esami_Colesterolo="si"))

    @Rule(AND(Fact(molto_dolore="si"), Fact(mancanza_fiato="si"), Fact(sente_disagio="si"), Fact(sente_sudorazione="si"), Fact(sente_debolezza="si")))
    def all_symptoms(self):
        print(Fore.RED + "Attenzione!!!!! Sembra che tu abbia TUTTI i sintomi di un attacco cardiaco")
        self.declare(Fact(tutti_sintomi="si"))
        self.declare(Fact(chiedi_esami_frequenzaCardiaca="si"))
        reset_color()

    @Rule(AND(Fact(diagnosi_frequenzaCardiaca_normale="si"), Fact(diagnosi_pressione_normale="si"),Fact(colesterolo_normale="si")))
    def not_symptoms(self):
        if self.number_prints == 0 and self.flag_no_symptoms == 1:
            print(Fore.GREEN + "Non hai alcun sintomo di attacco cardiaco ")
            self.declare(Fact(niente_sintomi="si"))
            reset_color()
            self.number_prints = self.number_prints + 1

    @Rule(NOT(AND(Fact(molto_dolore="si"), Fact(mancanza_fiato="si"), Fact(sente_disagio="si"), Fact(sente_sudorazione="si"), Fact(sente_debolezza="si"))))
    def not_symptoms2(self):
        if self.number_prints == 0 and self.flag_no_symptoms == 1:
            print(Fore.GREEN + "Non hai alcun sintomo di attacco cardiaco ")
            self.declare(Fact(niente_sintomi="si"))
            reset_color()
            self.number_prints = self.number_prints + 1



def main_agent():
    expert_agent = AttacchiDiCuore_expert()
    expert_agent.reset()
    expert_agent.run()



def main_ontology():
    do = AttaccoDiCuoreOntology()

    do.get_symptoms_descriptions()
    sympton, keys_sympton = do.print_symptoms()

    print("\nInserisci il numero del sintomo di cui vuoi conoscere la descrizione: ")
    numsintomo = int(input())

    print("Sintomo: %s \nDescrizione: %s"%(keys_sympton[numsintomo], " ".join(sympton[numsintomo])))


def main():

    exit_program = False

    print('\033[1m'+'\033[36m'+"SISTEMA DI DIAGNOSI AVVIATA"+'\033[0m')
    while exit_program == False:

        print("-----*MENU*------\n[1]Mostra i vari dati che influiscono la possibilita' di avere un attacco cardiaco\n[2]Calcola diagnosi\n[3]Esci")
        scelta = None

        try:
            scelta = int(input())

        except ValueError:
            exit_program = True

        if scelta == 1:
            main_ontology()

        elif scelta == 2:
            main_agent()

        else:
            print("Chiusura del programma...")
            exit_program = True


main()