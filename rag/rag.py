import re, math
from collections import Counter
from kb_data import PASSAGES

STOP = set("""a an the of to in on for and or is are was were be been what which who whom whose how why when where does do did
can could should would will with by at as from that this these those it its into than then there their about per
use used uses using value number many much tell me please give default typically exactly specifically configured
set according passage knowledge base kb system describe explain""".split())

def norm(t):
    t = t.lower().replace("’", "'")
    t = re.sub(r"(\d)\s+(ms|bits?)\b", r"\1\2", t)
    t = re.sub(r"[-_]", " ", t)
    return re.sub(r"\s+", " ", t)

def stem(w):
    for suf in ("ings","ing","ies","es","ed","s"):
        if len(w) > len(suf)+3 and w.endswith(suf):
            return w[:-len(suf)] + ("y" if suf=="ies" else "")
    return w

def toks(t):
    return [stem(w) for w in re.findall(r"[a-z0-9]+(?:\.\d+)?", norm(t)) if w not in STOP]

# (alias, weight) per passage
ALIAS = {
0:[("raft",1)],1:[("multi paxos",1),("paxos",1)],2:[("gossip",1),("phi accrual",1),("epidemic",1)],
3:[("swim",1)],4:[("two phase commit",1),("2pc",1),("three phase commit",1),("3pc",1)],
5:[("saga",1)],6:[("consistent hashing",1),("consistent hash",1),("hash ring",.7)],
7:[("rendezvous",1),("highest random weight",1),("hrw",1)],8:[("vector clock",1),("version vector",.8)],
9:[("hybrid logical clock",1),("hlc",1)],10:[("sgd",1),("stochastic gradient descent",1)],
11:[("adamw",1),("adam",.8)],12:[("transformer encoder",1),("encoder",.8)],
13:[("autoregressive",1),("decoder",.8),("causal",.6)],14:[("logistic regression",1)],
15:[("svm",1),("support vector",1)],16:[("mlp",1),("multi layer perceptron",1),("perceptron",1)],
17:[("resnet",1),("residual convolutional",1),("convolutional",.8),("cnn",.9)],
18:[("batch normalization",1),("batchnorm",1),("batch norm",1)],19:[("layer normalization",1),("layernorm",1),("layer norm",1)],
20:[("postgresql",1),("postgres",1),("pgbouncer",.8)],21:[("mysql",1),("innodb",1)],
22:[("b+ tree",1),("b tree",.9),("btree",.9)],23:[("lsm",1),("log structured merge",1),("sstable",.4)],
24:[("mongodb",1),("mongo",1)],25:[("cassandra",1)],26:[("redis",1)],27:[("memcached",1)],
28:[("write ahead log",1),("wal",1),("aries",.9)],29:[("shadow paging",1),("shadow page",1)],
30:[("tcp",.8),("transmission control protocol",1)],31:[("quic",1)],
32:[("dns",.7),("domain name system",1),("dnssec",.9)],33:[("mdns",1),("multicast dns",1),("bonjour",.8),("avahi",.8)],
34:[("http/2",1),("http 2",1),("http2",1),("hpack",1)],35:[("http/3",1),("http 3",1),("http3",1),("qpack",1)],
36:[("tls 1.3",1),("tls1.3",1),("transport layer security",.8),("tls",.6)],
37:[("tls 1.2",1),("tls1.2",1),("tls",.6)],38:[("bgp",1),("border gateway",1)],39:[("ospf",1),("open shortest path",1)],
40:[("public key",.9),("rsa",.8),("asymmetric",.6)],41:[("elliptic curve",1),("ecdsa",1),("ecc",.9),("ed25519",.8)],
42:[("password hashing",1),("credential storage",1),("argon2",.9),("salting",.9),("key stretching",.5)],
43:[("pbkdf2",1)],44:[("waf",1),("web application firewall",1),("firewall",.9)],
45:[("api gateway",1),("token bucket",1),("kong",.8)],46:[("aes",1),("advanced encryption standard",1)],
47:[("chacha20",1),("poly1305",1),("chacha",1)],48:[("pkce",1),("oauth",1),("proof key",1)],
49:[("jwt",1),("json web token",1)],
}
ALIAS_RE = {i:[(re.compile(r"(?<![a-z0-9])"+re.escape(norm(a))+r"s?(?![a-z0-9])"),w,norm(a)) for a,w in L] for i,L in ALIAS.items()}

DOCS = [toks(p) for p in PASSAGES]
N = len(DOCS); AVG = sum(map(len,DOCS))/N
DF = Counter(w for d in DOCS for w in set(d))
IDF = {w: math.log(1+(N-df+.5)/(df+.5)) for w,df in DF.items()}
TF = [Counter(d) for d in DOCS]

SD=[[Counter(toks(x)) for x in re.split(r"(?<=[a-z0-9\)])\.\s+(?=[A-Z])",p.rstrip("."))] for p in PASSAGES]
AVGS=sum(sum(c.values()) for L in SD for c in L)/sum(len(L) for L in SD)
def sent_bm25(q):
    qs=set(q); out=[]
    for L in SD:
        best=0
        for c in L:
            n=sum(c.values()); sc=0
            for w in qs:
                f=c.get(w,0)
                if f: sc+=IDF[w]*f*2.5/(f+1.5*(.25+.75*n/AVGS))
            best=max(best,sc)
        out.append(best)
    return out

def bm25(q):
    qs = Counter(q); out=[]
    for i in range(N):
        s=0; L=len(DOCS[i])
        for w in qs:
            f=TF[i].get(w,0)
            if f: s += IDF[w]*f*2.5/(f+1.5*(.25+.75*L/AVG))
        out.append(s)
    return out

INJ = re.compile(r"(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|instruction|rule|prompt|context)|system prompt|you are now|new instructions?|reveal .{0,20}(prompt|instruction|secret|key)|respond (only )?with|answer (only )?with|output (exactly|only)|pretend|jailbreak|developer mode|<\s*/?\s*(system|assistant|instruction)", re.I)

SYN = {"keepalive":["heartbeat"],"coordinator":["leader"],"subordinate":["follower"],"trigger":["timeout"],
 "scalar":["learning","rate"],"coefficient":["rate"],"adjustment":["update"],"periodic":["heartbeat"],
 "signal":["heartbeat"],"cryptographic":["crypto"],"socket":["port"],"duration":["ms"]}
def expand(ts):
    out=list(ts)
    for t in ts:
        for e in SYN.get(t,[]): out.append(stem(e))
    return out

PREM = re.compile(r"^\s*(since|given(?: that)?|assuming(?: that)?|because|considering(?: that)?|suppose(?: that)?|knowing(?: that)?|granted(?: that)?|as|if|now that|seeing (?:that|as))\b(.+?),\s*((?:what|which|how|when|where|who|why|does|do|is|are|can)\b.*)$", re.I|re.S)
def split_premise(q):
    m=PREM.match(q.strip())
    return (m.group(3).strip(), m.group(2)) if m else (q, "")

def strip_injection(q):
    q=re.sub(r"<!--.*?-->","",q,flags=re.S)
    q=re.sub(r"<[^>]{0,200}>","",q)
    parts = re.split(r"(?<=[.!?\n])\s+|\s*[\[\]{}]\s*|\s+---+\s+|\s*(?:\n|:)\s*(?=(?:note|system|instruction|assistant|important)\b)", q, flags=re.I)
    keep=[p for p in parts if p and not INJ.search(p)]
    out=" ".join(keep).strip()
    return out if out else q

# attribute: (name, query regex, sentence regex)
ATTR = [
 ("port", r"\bports?\b|listen|bind", r"\bport\b"),
 ("quorum", r"quorum|acknowledg|how many (nodes|acceptors|replicas|participants|members)", r"quorum"),
 ("hashfn", r"\bhash\b|digest|sha|blake|primitive|hash function|hashing algorithm", r"SHA-3|BLAKE2"),
 ("lr", r"learning rate|warm ?up|coefficient|scalar|step size", r"learning rate|warmup"),
 ("depth", r"\bblocks?\b|\blayers?\b|\bstack|\bdepth|\bdeep\b", r"\b(blocks|layers)\b"),
 ("bits", r"\bbits?\b|key (size|length)|\blength|modulus|curve size|key lengths?", r"\bbits?\b|-bit"),
 ("batch", r"batch|sample|chunk|records?|bytes|entries|instances|examples|window length|tokens per|per (step|frame|fsync)|group(ed|s)? into", r"samples|chunks|records|bytes|entries|tokens per|instances|examples|support vectors"),
 ("threshold", r"threshold|ratio|saturation|utilization|probability|burst|skew|capacity|metric|correlation|cutoff", r"\b0\.\d+"),
 ("time", r"time ?out|heartbeat|interval|how long|\bms\b|millisecond|timer|latency|window|deadline|expir|wait|lease|rto|keepalive|dead|penalty|duration|delay|period|ttl|clock skew|within", r"\d+ ?ms"),
]
ATTR_C = [(n,re.compile(q,re.I),re.compile(s,re.I)) for n,q,s in ATTR]
MASK = re.compile(r"TLS 1\.[23]|HTTP/[23]|SHA-3|BLAKE2|ChaCha20|Ed25519|Secp256r1|Poly1305|PBKDF2|OAuth 2\.0|Argon2|base64url|L2|b\+|RFC \d+|HTTP 429", re.I)
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(ms|bits?|-bit)?(?![\w])")

def sentences(p): return re.split(r"(?<=[a-z0-9\)])\.\s+(?=[A-Z])", p.rstrip("."))

def entity_scores(qn):
    sc={}
    for i,L in ALIAS_RE.items():
        best=0; spans=[]
        for r,w,a in L:
            if r.search(qn): best=max(best,w); spans.append(a)
        if best: sc[i]=(best,spans)
    # specific TLS versions beat generic tls
    if 36 in sc and 37 in sc:
        if sc[36][0]>sc[37][0]: sc.pop(37)
        elif sc[37][0]>sc[36][0]: sc.pop(36)
    return sc

def fmt(m):
    u=m.group(2) or ""
    if u in("bit","bits","-bit"): return m.group(1)+" bits"
    return m.group(1)+u

def extract_value(attr, sent, qterms):
    if attr=="hashfn":
        m=re.search(r"SHA-3|BLAKE2",sent); return m.group(0) if m else None
    masked=MASK.sub(lambda m:"#"*len(m.group(0)),sent)
    cands=[m for m in NUM.finditer(masked)]
    if not cands: return None
    if len(cands)==1: m=cands[0]; return fmt(m)
    words=[(m.start(),m.group(0)) for m in re.finditer(r"[A-Za-z]+",sent)]
    def dist(m):
        best=99
        for pos,w in words:
            if stem(w.lower()) in qterms or w.lower() in qterms:
                best=min(best,abs(pos-m.start())/6)
        return best
    m=min(cands,key=lambda m:(dist(m), 0 if m.group(2) else 1))
    return fmt(m)

def answer_from(i, qclean, qterms, attrs):
    sents=sentences(PASSAGES[i])
    chosen=[]
    for n,qr,sr in ATTR_C:
        if n in attrs:
            cands=[(k,s) for k,s in enumerate(sents) if sr.search(s)]
            if cands:
                k,s=max(cands,key=lambda ks:sum(1 for t in toks(ks[1]) if t in qterms))
                if (k,s,n) not in [(a,b,c) for a,b,c in chosen]: chosen.append((k,s,n))
    return chosen

MECH = re.compile(r"\bhow (does|do|did|can|would|could)\b|\bwhy\b|\bexplain\b|\bwalk me through\b|\bwhat happens\b")
FORMAL = re.compile(r"formulation|formula\b|equation|derivation|\bderive|mathematical|theorem|\bproof\b|pseudocode|source code")
CMP = re.compile(r"\b(compare|comparison|versus|vs|difference|differ|both|combined|sum|total|together|larger|smaller|higher|lower|greater|longer|shorter|bigger|more than|less than|ratio|contrast)\b")

def parse_num(v):
    m=re.match(r"\s*(\d+(?:\.\d+)?)",v or "")
    return float(m.group(1)) if m else None

def fmtn(x,u):
    return (str(int(x)) if x==int(x) else str(round(x,4)))+u

def compute(qn,items):
    # items: list of (name,value_str)
    vals=[(n,parse_num(v),re.sub(r"^[\d.]+","",v or "").strip()) for n,v in items]
    vals=[(n,x,u) for n,x,u in vals if x is not None]
    if len(vals)<2: return ""
    u=vals[0][2]; a,b=vals[0][1],vals[1][1]
    nm=lambda k:vals[k][0]
    if re.search(r"\b(sum|total|combined|together|add)\b",qn): return "Combined total: "+fmtn(a+b,u)+"."
    if re.search(r"\b(difference|differ|gap)\b",qn): return "Difference: "+fmtn(abs(a-b),u)+"."
    if re.search(r"\b(ratio)\b",qn) and b: return "Ratio "+nm(0)+":"+nm(1)+" = "+str(round(a/b,4))+"."
    if re.search(r"\b(larger|higher|greater|longer|bigger|more|larger|max|most)\b",qn):
        k=0 if a>=b else 1; return nm(k)+" is larger ("+fmtn(vals[k][1],u)+" vs "+fmtn(vals[1-k][1],u)+")."
    if re.search(r"\b(smaller|lower|shorter|less|fewer|min|least)\b",qn):
        k=0 if a<=b else 1; return nm(k)+" is smaller ("+fmtn(vals[k][1],u)+" vs "+fmtn(vals[1-k][1],u)+")."
    return ""

def ask(question, top_k=3):
    try: top_k=max(1,min(int(top_k),10))
    except Exception: top_k=3
    q=strip_injection(question or "")
    qpart,prem=split_premise(q)
    qn=norm(q)
    ent=entity_scores(qn)
    qt=expand(toks(qpart))
    qt_all=expand(toks(q))
    if ent: qt=qt+[w for i in ent for a in ent[i][1] for w in toks(a)]
    b1=bm25(qt); b2=sent_bm25(qt); bm=[x+1.5*y for x,y in zip(b1,b2)]; mx=max(bm) or 1
    bmn=[b/mx for b in bm]
    # entity-aware score
    sc=[]
    for i in range(N):
        e=ent.get(i,(0,[]))[0]
        s = 0.6*e+0.4*bmn[i] if ent else 0.8*bmn[i]
        sc.append(min(1.0,max(0.0,s)))
    order=sorted(range(N),key=lambda i:-sc[i])
    # strip entity words from query for attribute detection
    qclean=norm(qpart)
    for i,(w,spans) in ent.items():
        for a in spans: qclean=qclean.replace(a," ")
    qterms=set(expand(toks(qclean)))
    attrs=[n for n,qr,sr in ATTR_C if qr.search(qclean)]
    # abstention
    abst=False; top=order[0]
    maxbm=max(bm)
    multi_ent=[i for i,(w,_) in ent.items() if w>=0.8]
    mech=bool(MECH.search(qn)); formal=bool(FORMAL.search(qn))
    def covfn(t,terms):
        dset=set(DOCS[t])
        tot=sum(IDF.get(w,3.0) for w in terms) or 1
        return sum(IDF.get(w,3.0) for w in terms if w in dset)/tot
    if len(multi_ent)>=2 or CMP.search(qn):
        pass
    elif not ent:
        terms=set(qt)
        if formal: abst=True
        elif attrs:
            if maxbm<14 or len(terms)<2: abst=True
        else:
            if mech or covfn(top,terms)<0.5 or maxbm<14 or len(terms)<2: abst=True
    else:
        top=max(ent,key=lambda i:(ent[i][0],sc[i]))
        terms=set(qterms)
        cov=covfn(top,terms) if terms else 1.0
        if formal: abst=True
        elif mech and cov<0.75: abst=True
        elif cov<0.25 and not attrs: abst=True
    chosen=[]
    multi=multi_ent
    if not abst:
        if len(multi)>=2: targets=sorted(multi,key=lambda i:-sc[i])[:3]
        elif CMP.search(qn):
            base=top if ent else order[0]
            others=[i for i in order if i!=base and bmn[i]>=0.35]
            targets=[base]+others[:1]
        else: targets=[order[0] if not ent else top]
        parts=[]
        for t in targets:
            ch=answer_from(t,qclean,qterms,attrs)
            if attrs and not ch and len(targets)==1:
                abst=True; break
            if not ch:
                sents=sentences(PASSAGES[t])
                if len(targets)==1 and ent:
                    # require some overlap to avoid hallucinated answers
                    k,s=max(enumerate(sents),key=lambda ks:sum(IDF.get(w,0) for w in set(toks(ks[1])) if w in qterms))
                    ov=sum(IDF.get(w,0) for w in set(toks(s)) if w in qterms)
                    if qterms and ov==0: abst=True; break
                    ch=[(k,s,None)]
                elif len(targets)==1:
                    ch=[(0,sentences(PASSAGES[t])[0],None)]
                else:
                    ch=[(0,sentences(PASSAGES[t])[0],None)]
            parts.append((t,ch))
        if not abst:
            texts=[]; comp=[]
            for t,ch in parts:
                seen=set(); seg=[]
                for k,s,n in sorted(ch,key=lambda c:c[0]):
                    v=extract_value(n,s,qterms) if n else None
                    seg.append((v,s))
                vals=[v for v,_ in seg if v]
                comp.append((ALIAS[t][0][0].title() if t in ALIAS else str(t), vals[0] if vals else None))
                body=" ".join(s.rstrip(".")+"." for _,s in seg)
                lead=(", ".join(dict.fromkeys(vals))+". ") if vals else ""
                if len(parts)>1:
                    name=ALIAS[t][0][0].title() if t in ALIAS else str(t)
                    texts.append(f"{name}: {lead}{body}")
                else: texts.append(lead+body)
            answer=" | ".join(texts) if len(parts)>1 else texts[0]
            if len(parts)>1:
                c=compute(qn,comp)
                if c: answer=answer+" | "+c
            if len(parts)>1:
                # float tied targets to the front of the ranking
                order=[t for t,_ in parts]+[i for i in order if i not in [t for t,_ in parts]]
            else:
                order=[top]+[i for i in order if i!=top]
    if abst:
        return {"answer":None,"sources":[],"scores":[],"confidence":0.1}
    src=order[:top_k]
    scores=sorted([round(max(0.0,min(1.0,sc[i])),4) for i in src],reverse=True)
    # ensure primary source has high score
    scores=[max(s,0.0) for s in scores]
    conf=round(min(0.97,max(0.65,0.6+0.4*scores[0])),3)
    return {"answer":answer,"sources":src,"scores":scores,"confidence":conf}
