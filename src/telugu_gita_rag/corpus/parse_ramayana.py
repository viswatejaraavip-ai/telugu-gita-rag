import re, json, urllib.parse
from build_corpus import wikitext, strip_markup, allpages

TE=str.maketrans("౦౧౨౩౪౫౬౭౮౯","0123456789")
PIPE=re.compile(r"\|\s*([\d౦-౯]{1,2})\s*[-–]\s*([\d౦-౯]{1,3})\s*[-–]\s*([\d౦-౯]{1,3})([a-z]?)\s*\|"
                 r"|\|\|\s*(\d{1,2})\.(\d{1,3})\.(\d{1,3})()")
BRK =re.compile(r"\[\s*([\d౦-౯]{1,2})\s*[-–]\s*([\d౦-౯]{1,3})\s*[-–]\s*([\d౦-౯]{1,3})[a-z]?\s*\]")
KANDAS=[("బాలకాండము",1),("అయోధ్యాకాండము",2),("అరణ్యకాండము",3),("కిష్కింధకాండము",4),
        ("సుందరకాండము",5),("యుద్ధకాండము",6),("ఉత్తరకాండము",7)]

def parse_sarga(title,kanda,kno):
    body=strip_markup(wikitext(title))
    m=re.search(r"సర్గము\s*(\d+)",title)
    sarga=int(m.group(1)) if m else 0
    gloss={}
    for g in BRK.finditer(body):
        k,s,v=[int(x.translate(TE)) for x in g.groups()]
        seg=body[:g.start()].split("|")[-1]
        gloss[(s,v)]=re.sub(r"\s+"," ",seg).strip()
    out=[];prev=0
    for mm in PIPE.finditer(body):
        g=[x for x in mm.groups() if x is not None]
        if len(g)<3: continue
        k,s,v=g[0],g[1],g[2]
        try: s2,v2=int(s.translate(TE)),int(v.translate(TE))
        except: continue
        sl=re.sub(r"\s+"," ",body[prev:mm.start()]).strip(" |।॥")
        prev=mm.end()
        if len(sl)<12 or len(sl)>1200: continue
        out.append(dict(id=f"ram-{kno}-{s2}-{v2}",text_type="verse",
            work="Valmiki Ramayana",work_te="వాల్మీకి రామాయణము",
            kanda=kanda,kanda_no=kno,sarga_no=s2,sloka_no=v2,
            citation_en=f"Valmiki Ramayana {kno}.{s2}.{v2}",
            citation_te=f"వాల్మీకి రామాయణము, {kanda}, సర్గ {s2}, శ్లోకం {v2}",
            sloka_sa_in_telugu_script=sl,
            tatparyam_te=gloss.get((s2,v2),""),
            has_tatparyam=bool(gloss.get((s2,v2))),
            source_page=title,
            source_url="https://te.wikisource.org/wiki/"+urllib.parse.quote(title.replace(" ","_")),
            license="CC BY-SA 4.0"))
    return out

if __name__=="__main__":
    allv=[]
    for kanda,kno in KANDAS:
        pages=allpages(f"{kanda} - సర్గము")+allpages(f"{kanda}/సర్గము")
        n=0
        for t in pages:
            v=parse_sarga(t,kanda,kno); allv+=v; n+=len(v)
        print(f"  {kanda:<18} {len(pages):>4} sargas  {n:>6} slokas")
    with open("corpus/ramayana.jsonl","w",encoding="utf-8") as f:
        for r in allv: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    g=sum(1 for r in allv if r["has_tatparyam"])
    print(f"\nramayana.jsonl: {len(allv)} slokas, {g} with Telugu tatparyam ({100*g/len(allv):.1f}%)")
