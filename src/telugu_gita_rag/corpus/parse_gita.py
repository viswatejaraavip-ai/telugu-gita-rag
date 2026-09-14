import re, json, urllib.parse
from build_corpus import wikitext, strip_markup

GREF = re.compile(r"\|\|\s*(\d{1,2})\s*[-–]\s*(\d{1,3})\s*\|\|")
TE   = re.compile(r"[ఀ-౿]")
DEVA = re.compile(r"[ऀ-ॿ]")
CHAPTERS=[(1,"అర్జునవిషాద యోగము"),(2,"సాంఖ్య యోగము"),(3,"కర్మ యోగము"),(4,"జ్ఞాన యోగము"),
(5,"కర్మసన్యాస యోగము"),(6,"ఆత్మసంయమ యోగము"),(7,"జ్ఞానవిజ్ఞాన యోగము"),(8,"అక్షరపరబ్రహ్మ యోగము"),
(9,"రాజవిద్యారాజగుహ్య యోగము"),(10,"విభూతి యోగము"),(11,"విశ్వరూపసందర్శన యోగము"),(12,"భక్తి యోగము"),
(13,"క్షేత్రక్షేత్రజ్ఞవిభాగ యోగము"),(14,"గుణత్రయవిభాగ యోగము"),(15,"పురుషోత్తమప్రాప్తి యోగము"),
(16,"దైవాసురసంపద్విభాగ యోగము"),(17,"శ్రద్దాత్రయవిభాగ యోగము"),(18,"మోక్షసన్యాస యోగము")]
EXPECTED={1:47,2:72,3:43,4:42,5:29,6:47,7:30,8:28,9:34,10:42,11:55,12:20,
          13:35,14:27,15:20,16:24,17:28,18:78}

def rec(ch,name,vs,sloka,tat,mode):
    return dict(id=f"gita-{ch}-{vs}", text_type="verse", work="Bhagavad Gita",
        work_te="భగవద్గీత", adhyaya_no=ch, adhyaya=name, sloka_no=vs,
        citation_en=f"Bhagavad Gita {ch}.{vs}",
        citation_te=f"భగవద్గీత {ch}.{vs} ({name})",
        sloka_sa=re.sub(r"\s+"," ",sloka).strip(" |।॥"),
        sloka_is_incipit_only=(mode=="B"),
        tatparyam_te=re.sub(r"\s+"," ",tat).strip(),
        source_page=name,
        source_url="https://te.wikisource.org/wiki/"+urllib.parse.quote(name.replace(" ","_")),
        license="CC BY-SA 4.0")

def parse_A(body, ch, name):
    """full sloka + '|| c-v ||' + tatparyam"""
    out, prev_end, pend = [], 0, None
    for m in GREF.finditer(body):
        seg=[l.strip() for l in body[prev_end:m.start()].split("\n") if l.strip()]
        h=None
        for i in range(len(seg)-1):
            if len(seg[i])>=8 and seg[i+1].startswith(seg[i][:8]): h=i; break
        tat  = " ".join(seg[:h]) if h is not None else " ".join(seg[:-1])
        slok = " ".join(seg[h+1:]) if h is not None else (seg[-1] if seg else "")
        if pend and len(TE.findall(tat))>=8:
            pend["tatparyam_te"]=re.sub(r"\s+"," ",tat).strip(); out.append(pend)
        pend = rec(int(m.group(1)), name, int(m.group(2)), slok, "", "A")
        prev_end=m.end()
    if pend:
        seg=[l.strip() for l in body[prev_end:].split("\n") if l.strip()]
        tat=" ".join(seg[:-1]) if len(seg)>1 else " ".join(seg)
        if len(TE.findall(tat))>=8:
            pend["tatparyam_te"]=re.sub(r"\s+"," ",tat).strip(); out.append(pend)
    return out

def parse_B(body, ch, name):
    """Short Sanskrit incipit line starts a verse; following Telugu prose lines are its gloss."""
    lines=[l.strip() for l in body.split("\n") if l.strip()]
    verses, cur = [], None
    for l in lines:
        is_incipit = len(l) < 55 and not l.rstrip().endswith((".", "?", "!", "\u0964"))
        if is_incipit:
            if cur and len(TE.findall(cur[1]))>=15: verses.append(cur)
            cur=[l,""]
        elif cur is not None:
            cur[1]=(cur[1]+" "+l).strip()
    if cur and len(TE.findall(cur[1]))>=15: verses.append(cur)
    return [rec(ch,name,i+1,inc,tat,"B") for i,(inc,tat) in enumerate(verses)]

def parse_chapter(ch,name):
    from mode_c import detect as detect_C, parse_C
    body=strip_markup(wikitext(name))
    body=re.sub(r"భాష్యాలు\s*:.*?(?=\n\s*\n[^\s])"," ",body,flags=re.S)   # drop commentaries
    body=re.sub(r"__[A-Z]+__"," ",body)
    if detect_C(body):
        return parse_C(body,ch,name,rec), "C"
    if GREF.search(body):
        return parse_A(body,ch,name), "A"
    return parse_B(body,ch,name), "B"

if __name__=="__main__":
    allv=[]; print(f"{'ch':>3} {'name':<30}{'mode':>5}{'got':>5}{'exp':>5}  ok")
    for n,name in CHAPTERS:
        v,mode=parse_chapter(n,name); allv+=v
        ok="OK" if abs(len(v)-EXPECTED[n])<=2 else "** CHECK"
        print(f"{n:>3} {name:<30}{mode:>5}{len(v):>5}{EXPECTED[n]:>5}  {ok}")
    print(f"\nTOTAL {len(allv)} / {sum(EXPECTED.values())}")
    with open("corpus/gita.jsonl","w",encoding="utf-8") as f:
        for r in allv: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    for i in (60,300):
        print(f"\n--- sample {allv[i]['id']} ---")
        print(json.dumps(allv[i],ensure_ascii=False,indent=2))
