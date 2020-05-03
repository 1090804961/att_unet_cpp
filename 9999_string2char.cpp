//
// Created by ZY on 2020-05-03.
//

1 string to CString   

  CString.format("%s",string.c_str()); 

2 CString to string

        string str(CString.GetBuffer(str.GetLength()));

3 string to char *

char *p=string.c_str();

4 char * to string

        string str(char*);

5 CString to char *

strcpy(char,CString,sizeof(char));

6 char * to CString

        CString.format("%s",char*);
