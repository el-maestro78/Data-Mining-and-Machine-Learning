import pip


def check_import(libs_list):
    for library in libs_list:
        try:
            __import__(library)
        except:
            print(f"{library} is not imported")
            pip.main(["install", library])
        else:
            print(f"{library} is imported")


libraries = ["numpy", "matplotlib", "pandas"]
check_import(libraries)
