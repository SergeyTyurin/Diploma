import os

def writeEnvParam(bash_data,envName,envParam):
    lines = bash_data
    index=0
    for line in lines:
        if line.find(envName)!=-1:
            lines[index] = "export {0}={1}\n".format(envName,envParam)
            return lines
        index+=1
    lines.append("export {0}={1}\n".format(envName,envParam))
    return lines

try:
    bashfile = open("{0}/.bash_profile".format(os.getenv("HOME")),'r+')
    envDiplomaDIR = os.getcwd()
    envLMDBDataSetDIR = os.path.abspath(os.path.join(envDiplomaDIR,"..","DataSet","lmdb"))
    envTestDataSetDIR = os.path.abspath(os.path.join(envDiplomaDIR, "..", "DataSet", "Destination","Test"))
    bash_data = bashfile.readlines()
    bash_data = writeEnvParam(bash_data,"DiplomaDIR",envDiplomaDIR)
    bash_data = writeEnvParam(bash_data,"LMDBDataSetDIR",envLMDBDataSetDIR)
    bash_data = writeEnvParam(bash_data,"TestDataSetDIR",envTestDataSetDIR)
    bashfile.seek(0)
    bashfile.writelines(bash_data)
except Exception as e:
    print(e)