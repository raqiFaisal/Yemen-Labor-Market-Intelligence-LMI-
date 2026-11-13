"""Offline Streamlit page-logic smoke tests with actual project data/models.

This is NOT a browser render test. It substitutes Streamlit widgets with an
instrumented stand-in because Streamlit is not installed in the audit runtime.
Every page Python body is executed; specified button paths are exercised.
"""
from __future__ import annotations
import json, runpy, sys, functools, types, traceback
from pathlib import Path
from src.common import ROOT, REPORTS_DIR, write_json

class StopPage(Exception): pass

class Context:
    def __init__(self,st):self.st=st
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def __getattr__(self,key):return getattr(self.st,key)

class StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.sidebar=Context(self)
        self.column_config=types.SimpleNamespace(LinkColumn=lambda *args,**kwargs:{'kind':'link','args':args})
        self.pressed=set();self.overrides={};self.calls=[];self.notifications=[];self.pages=[]
    def configure(self, pressed=(), overrides=None):
        self.pressed=set(pressed); self.overrides=overrides or {};self.calls=[];self.notifications=[]
    def _take(self,method,*args,**kwargs):
        self.calls.append((method, str(args[0])[:70] if args else ''))
        if method in ('warning','error','info','success'):self.notifications.append((method,str(args[0])[:200]))
    def cache_resource(self,*args,**kwargs):
        return lambda fn: functools.lru_cache(maxsize=8)(fn)
    def columns(self,n,*args,**kwargs):
        self._take('columns',n)
        return [Context(self) for _ in range(n if isinstance(n,int) else len(n))]
    def tabs(self,options,*args,**kwargs):
        self._take('tabs', options)
        return [Context(self) for _ in options]
    def expander(self,*args,**kwargs):return Context(self)
    def spinner(self,*args,**kwargs):return Context(self)
    def button(self,label,*args,**kwargs):
        self._take('button',label)
        return label in self.pressed
    def radio(self,label,options,*args,**kwargs):
        self._take('radio',label)
        return self.overrides.get(label, options[0])
    def selectbox(self,label,options,*args,**kwargs):
        self._take('selectbox',label)
        return self.overrides.get(label, options[0] if len(options) else None)
    def select_slider(self,label,*args,**kwargs):
        self._take('select_slider',label)
        return self.overrides.get(label, kwargs.get('value'))
    def multiselect(self,label,options,*args,**kwargs):
        self._take('multiselect',label)
        return self.overrides.get(label,kwargs.get('default',[]))
    def text_area(self,label,*args,**kwargs):
        self._take('text_area',label)
        return self.overrides.get(label,'')
    def text_input(self,label,*args,**kwargs):
        self._take('text_input',label)
        return self.overrides.get(label,'')
    def Page(self,path,*args,**kwargs):
        self.pages.append(path)
        return types.SimpleNamespace(path=path, title=kwargs.get('title'))
    def navigation(self,pages,*args,**kwargs):
        return types.SimpleNamespace(run=lambda: self._take('navigation.run'))
    def stop(self):raise StopPage()
    def download_button(self,label,data,*args,**kwargs):
        self._take('download_button',label)
        if not data:raise AssertionError('Empty downloadable CSV')
        return True
    def plotly_chart(self, fig, **kwargs):
        self._take('plotly_chart')
        if fig is None:raise AssertionError('chart missing')
    def __getattr__(self,name):
        if name.startswith('__'):raise AttributeError(name)
        def fun(*args,**kwargs):
            self._take(name,*args,**kwargs)
            return None
        return fun

ST=StreamlitStub()
sys.modules['streamlit']=ST
sys.path.insert(0,str(ROOT/'app_dl'))

CASES=[
 ('main_navigation','main', (), {}),
 ('dashboard','dashboard',(),{}),
 ('market_default','market',(),{}),
 ('market_technical','market',(),{'نوع المهارة':'technical','المقياس':'Lift'}),
 ('search_all','search',(),{}),
 ('search_query','search',(),{'كلمات البحث':'محاسب'}),
 ('analyze_unpressed','analyze',(),{}),
 ('analyze_valid','analyze',('تحليل',),{'نص الإعلان':'تعلن شركة تقنية في صنعاء عن وظيفة مهندس برمجيات تتطلب Python و SQL وخبرة سنتين.'}),
 ('analyze_empty','analyze',('تحليل',),{}),
 ('gap_unpressed','gap',(),{}),
 ('gap_valid','gap',('حلّل الفجوة',),{'الهدف':'كل السوق','مهارات أخرى (افصل بفاصلة)':'Excel'}),
 ('gap_empty','gap',('حلّل الفجوة',),{}),
 ('gap_title','gap',('حلّل الفجوة',),{'الهدف':'مسمى وظيفي','المسمى':'محاسب','مهارات أخرى (افصل بفاصلة)':'Excel'}),
 ('models','models',(),{}),
 ('about','about',(),{}),
]+[(f'forecast_{name}','forecast',(),{'العائلة':name}) for name in ('skills','specializations','job_titles','governorates')]

def run():
    result=[]
    for name,page,pressed,overrides in CASES:
        ST.configure(pressed,overrides)
        status='pass';details=''
        try:
            runpy.run_path(str(ROOT/'app_dl'/('main.py' if page=='main' else f'views/{page}.py')),run_name=f'ui_test_{name}')
        except StopPage:
            status='expected_stop' if name in ('analyze_empty','gap_empty') else 'unexpected_stop'
        except Exception as ex:
            status='fail';details=traceback.format_exc(limit=6)
        if name=='main_navigation' and len(ST.pages)!=8:
            status='fail';details=f'expected 8 navigation pages; got {len(ST.pages)}'
        result.append({'case':name,'page':page,'status':status,'widgets_called':len(ST.calls),
                       'notifications':ST.notifications[:8],'details':details})
        print(name,status,'widgets=',len(ST.calls), details[:180])
    out={'method':'Static Python page execution with simulated Streamlit controls, actual engine/data/checkpoints; NOT browser/UI rendering',
         'passed':sum(x['status'] in ('pass','expected_stop') for x in result), 'total':len(result),
         'cases':result, 'all_passed':all(x['status'] in ('pass','expected_stop') for x in result)}
    dest=REPORTS_DIR/'evaluation'/'ui_logic_smoke.json'
    write_json(dest,out)
    return out
if __name__=='__main__':
    r=run();print('SUMMARY',r['passed'],'/',r['total']);
    if not r['all_passed']:raise SystemExit(1)
