import re, random

class DataReader:
   '''This class supports a functionality to read in data from the data files.
   the class supports iteration and can be used in a for each loop'''
   def __init__(self, dataFile):
      '''Initialize the DataReader.  Pass in a datafile'''
      self.f = open(dataFile, "r")
       
   def __iter__(self):
      return self

   def next(self):
      ''' Returns the next label/data pair from the data file'''
      self.readNext()

      if self.data == "":
         raise StopIteration
      else:
         return (self.label, self.data)

   def readNext(self):
      '''Read and tokenize the next labal/data pair from the file'''
      self.data = ""
      self.label = ""

      for line in self.f:
         line = line.strip()
         match = re.match(r"<LABEL>(.*)<\/LABEL>", line)

         if match:
            self.label = match.group(1)
         elif line == "</DOC>":
            if self.data != "" and self.label != "":
               self.data = tokenize(self.data)
               break
            else:
               print "Warning: found empty doc"
         elif line == "<DOC>":
            # the start of a new document
            self.data = ""
            self.label = ""
         else:
            self.data += line + "\n"
            
def tokenize(sText):
   '''Given a string of text sText, returns a list of the individual tokens that 
   occur in that string (in order).'''

   lTokens = []
   sText = sText.lower().strip()
   sToken = ""
   for c in sText:
      if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
         sToken += c
      else:
         if sToken != "":
            lTokens.append(sToken)
            sToken = ""
         if c.strip() != "":
            lTokens.append(str(c.strip()))
               
   if sToken != "":
      lTokens.append(sToken)

   return lTokens

def split(dataFile, outputLabel):   
   '''Splits data into 80% train, 10% dev and 10% test'''
   reader = DataReader(dataFile)

   train = open(outputLabel + ".train", "w")
   dev = open(outputLabel + ".dev", "w")
   test = open(outputLabel + ".test", "w")

   for label, tokens in reader:
      output = "<DOC>\n<LABEL>" + label + "</LABEL>\n" + " ".join(tokens) + "\n</DOC>\n"

      # random selection
      selection = random.randint(1,10)

      if selection == 1:
         dev.write(output)
      elif selection == 2:
         test.write(output)
      else:
         train.write(output)
         
   train.close()
   dev.close()
   test.close()
