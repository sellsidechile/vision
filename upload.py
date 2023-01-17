import gspread
from oauth2client.service_account import ServiceAccountCredentials

class GoogleSheet:
    def __init__(self):
        scope = ['https://www.googleapis.com/auth/drive']
        
        creds = ServiceAccountCredentials.from_json_keyfile_name('Auth/Cred.json', scope)
        client = gspread.authorize(creds)
        sheet = client.open('EVENTO SOBREMESA')
        self.peoplecount = sheet.worksheet('CAMARA')
        self.lenLeft = 0
        self.lenRight = 0
        
    def getRowLength(self):
        return len(self.peoplecount.get_all_values())

    def getRow(self):
        return (self.peoplecount.get_all_values())

    def lengthLeftRigth(self):
        data = self.peoplecount.get_all_values()
        for row in data:
            if row[0] == 'Left':
                self.lenLeft += 1
            elif row[0] == 'Rigth':
                self.lenRight += 1
        
        return self.lenLeft, self.lenRight

    def sendData(self, Registo, Hora):
        data = []
        numId = self.getRowLength()

        data.append(Registo)
        data.append(Hora)

        self.peoplecount.insert_row(data, numId + 1)