import PySimpleGUI as sg
import google.generativeai as genai
from decouple import config

api_key = config("API_GEMINI")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


def answer_gemini(question):
    response = model.generate_content(question)
    return response.text


class Tela:
    def __init__(self):
        self.themes = sg.theme_list()
        self.create_window()

    def create_window(self, theme=None):
        sg.theme("Black")
        if theme is not None:
            sg.theme(theme)
        layout = [
            [
                sg.Frame(
                    "",
                    layout=[
                        [sg.Text("Escolha um tema:")],
                        [
                            sg.Combo(
                                self.themes,
                                default_value="Default",
                                key="-THEME-",
                                enable_events=True,
                            )
                        ],
                        [sg.Text("Digite sua pergunta:")],
                        [sg.InputText(key="-QUESTION-")],
                        [sg.Button("Submit"), sg.Button("Limpar"), sg.Exit()],
                        [sg.Text("Respostas:", size=(150, 1))],
                        [sg.Output(size=(200, 30), key="-OUTPUT-")],
                    ],
                    relief=sg.RELIEF_SUNKEN,
                )
            ]
        ]
        self.window = sg.Window("Q&A Interface", layout)

    def run(self):
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED or "Exit" in event:
                sg.popup_auto_close("Saindo...", auto_close_duration=0.5)
                break
            elif event == "-THEME-":
                self.window["-QUESTION-"].update("")
                new_theme = sg.theme(values["-THEME-"])
                print(new_theme)
                self.window.close()
                self.create_window(new_theme)
            elif "Limpar" in event:
                self.window["-QUESTION-"].update("")
            elif "Submit" in event:
                question = values["-QUESTION-"]
                if question:
                    responses = answer_gemini(question)
                    self.window["-OUTPUT-"].update(responses)
                else:
                    self.window["-OUTPUT-"].update("Digite uma pergunta")
        self.window.close()


if "__main__" == __name__:
    tela = Tela()
    tela.run()
