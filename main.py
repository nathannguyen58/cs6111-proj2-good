import pprint
import sys
import utils
from googleapiclient.discovery import build
'''
main function:
takes input from the user in the form of <api key>, <engine id>, <r>, <t>, <q>, <k>
'''

def main():
    args = sys.argv[1:]
    service = build("customsearch", "v1", developerKey = args[0])
    res = service.cse().list(q = args[4], cx = args[1],).execute()

    utils.processQuery(service, args[0], args[1], args[2], args[3], args[4], args[5])

if __name__ == "__main__":
    main()
