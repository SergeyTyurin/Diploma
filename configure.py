import os

try:
    bashfile = open("{0}/.bash_profile".format(os.getenv("HOME")),'a+')
    envDiplomaDIR = os.getcwd()
    envLMDBDataSetDIR = os.path.abspath(os.path.join(envDiplomaDIR,"..","DataSet","lmdb"))
    envTestDataSetDIR = os.path.abspath(os.path.join(envDiplomaDIR, "..", "DataSet", "Destination","Test"))

    bashfile.write("export DiplomaDIR={0}\n".format(envDiplomaDIR))
    bashfile.write("export LMDBDataSetDIR={0}\n".format(envLMDBDataSetDIR))
    bashfile.write("export TestDataSetDIR={0}\n".format(envTestDataSetDIR))
except Exception as e:
    print(e)