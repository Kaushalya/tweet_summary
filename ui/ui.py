import Tkinter
from tkFileDialog import *
import tkMessageBox
import summarizer as summ

root = Tkinter.Tk()
root.title("Tweet Summarizer")
modelFileText = Tkinter.StringVar()
dataFileText = Tkinter.StringVar()
outputFileText = Tkinter.StringVar()
statusText = Tkinter.StringVar()
model = None

def quitCallback():
   tkMessageBox.showinfo( "Hello Python", "Hello World")

def openCallback(textVar):
    fileName = askopenfilename(parent=root)
    textVar.set(fileName)
    print fileName
    
def saveCallback(textvar):
    fileName = asksaveasfilename(parent=root)
    textvar.set(fileName)
    print fileName    

def loadCallback():
    global model
    model=summ.load_model(modelFileText.get())
    
def summarizeCallback():
    statusText.set("summarizing...")
    status = summ.summarize_text(model, dataFileText.get(), outputFileText.get())
    statusText.set(status)
    
def addMenu():
    menubar = Tkinter.Menu(root)
    fileMenu = Tkinter.Menu(menubar, tearoff=0)
    fileMenu.add_command(label="Open", command=openCallback)
    fileMenu.add_checkbutton(label="train")
    fileMenu.add_checkbutton(label="test")
    fileMenu.add_separator()
    fileMenu.add_command(label="Exit", command=quitCallback)
    menubar.add_cascade(label="File", menu=fileMenu)
    root.config(menu=menubar)

def addSummPanel():
    topframe = Tkinter.LabelFrame(root, text="Load model", height=400, width=200, relief=Tkinter.SUNKEN, 
                             bg="", colormap="new", bd=1)
                             
    bottomframe = Tkinter.LabelFrame(root, text="Load data", height=600, width=200, relief=Tkinter.SUNKEN, 
                             bg="", colormap="new", bd=1)
                             
    topframe.grid(row=1, padx=5, pady=5, columnspan=3) 
    bottomframe.grid(row=2, padx=5, pady=5, columnspan=3) 
    
    Tkinter.Label(topframe, text="Model", anchor=Tkinter.W).grid(row=1, column=0, padx=5, pady=5)
    Tkinter.Button(topframe, text="Load", command=loadCallback).grid(row=2, column=1, padx=5, pady=5)    
    
    Tkinter.Label(bottomframe, text="Data").grid(row=5, column=0)
    Tkinter.Label(bottomframe, text="Output").grid(row=6, column=0)    
    
    Tkinter.Entry(topframe, textvariable=modelFileText).grid(row=1, column=1)
    Tkinter.Entry(bottomframe, textvariable=dataFileText).grid(row=5, column=1)
    Tkinter.Entry(bottomframe, textvariable=outputFileText).grid(row=6, column=1)
    
    Tkinter.Button(topframe, text="Browse...", command=lambda:openCallback(modelFileText)).grid(row=1, column=2)
    Tkinter.Button(bottomframe, text="Browse...", command=lambda:openCallback(dataFileText)).grid(row=5, column=2)
    Tkinter.Button(bottomframe, text="Browse...", command=lambda:saveCallback(outputFileText)).grid(row=6, column=2)
    
    Tkinter.Button(bottomframe, text="Summarize", command=summarizeCallback).grid(row=8, column=1)
    Tkinter.Button(root, text="Exit", command=quitCallback).grid(row=9, column=1)
    
    statusLabel = Tkinter.Label(root, textvariable=statusText)
    statusText.set("...")
    statusLabel.grid(row=10, column=1)

if __name__ == '__main__':
#    w = Tkinter.Label(root, textvariable=labelText) 
#    w.grid(row=1)
    addMenu()
    addSummPanel()
    
    root.mainloop( )