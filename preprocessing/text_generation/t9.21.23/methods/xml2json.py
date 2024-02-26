
import xmltodict
import pdb
xml_str = """
 <xml>
<ToUserName><![CDATA[gh_866835093fea]]></ToUserName>
<FromUserName><![CDATA[ogdotwSc_MmEEsJs9-ABZ1QL_4r4]]></FromUserName>
<CreateTime>1478317060</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[你好]]></Content>
<MsgId>6349323426230210995</MsgId>
</xml>
"""
 
xml_dict = xmltodict.parse(xml_str)
print(xml_dict)
print(type(xml_dict))
pdb.set_trace()