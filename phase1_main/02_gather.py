a
    ?l?a?  ?                   @   sv  d dl Z d dlmZ d dlmZ g d?Ze ?d?	dd? eD ??? g d?Z
ed?	d	d? e
ddd
? D ????jZg d?Zd?	dd? eddd
? D ??Zejeed?d? e?d?Ze?d?Zdd? eD ?ZW d  ? n1 s?0    Y  ed? dZe ?e? ed? d ZdD ?]8Zed7 Zeee?d ? dd? eD ?Zedk?rHdZndZde d e d ZeD ?]?Ze?d?d Ze?d?d  ?d!?d  Ze?d?d
 Zed"k?r?ee d# e d e d Z n?ed$v ?r?ee d% e d e d& Z n?ed'v ?r*ed(k?rd)Z!neZ!ee d* e d e! d+ Z nVedk?r?ed,k?rDd-Z"nd.Z"ed/k?rXd0Z!neZ!ee d* e d e" d e! d+ Z d1e d e Z#ee#ddd
? ?$? ??%? d d2? ddd
? Z&d1e d e d e Z'e'?d3?d  Z(ee(ddd
? ?$? ??%? d4d5? ddd
? Z)d6e& d7 e) d8 Z*d9e  e d: e* Ze ?e? ?qd?qg d;?Ze ?d?	d<d? eD ??? ed=? dS )>?    N)?md5)?ZipFile)?c   ?h   ?m   ?o   ?d   ?    ?a   ?+   ?x   r	   ?.   ?b   ?l   ?i   ?z   r   r
   ?r   r   ? c                 C   s   g | ]}t |??qS ? ??chr??.0?xr   r   ?02_gather.py?
<listcomp>   ?    r   )?p   r   r   r   r   r   ?s   r   ?/   r   r   r
   r   r   r   r   r   r   c                 C   s   g | ]}t |??qS r   r   r   r   r   r   r      r   ?????)	?T   ?A   ?E   ?H   ?C   r!   ?N   ?O   ?D   c                 C   s   g | ]}t |??qS r   r   r   r   r   r   r      r   zutf-8)?pwdz
gather.scp?   
c                 C   s   g | ]}|? d ??qS )?ascii)?decoder   r   r   r   r      r   z  ===== Gathering samples ===== 
zmkdir -p gatheredz gathering data, please wait.....)?2008?2009?2010?2011?2013?2016?   z/6c                 C   s&   g | ]}|? d ?d dt kr|?qS )?-r   ?BC)?split?y)r   ?kr   r   r   r   %   r   r1   ?2?1z#.blizzard/blizzard_wavs_and_scores_Z_release_version_?/r4   ?   ?_r-   z#/submission_directory/english/full/)r.   r/   z"/submission_directory/english/EH1/z/wavs/)r0   r2   r2   Z	audiobookz/submission_directory/z/wav/?BzEH2-EnglishzEH1-EnglishZbooksentZaudiobook_sentencesr5   ?   ?.?   ?   ?sysz-uttz.wavzcp z
 gathered/)r   r   r   r   r   r	   r
   ?-   r   r	   r   r   r   r   r   r   r
   r   r   c                 C   s   g | ]}t |??qS r   r   r   r   r   r   r   T   r   Zdone)+?osZhashlibr   ZhlmZzipfiler   Zzf?cmd?system?join?a?z?o?pZsetpassword?bytes?read?fr6   ?lZkl?print?cr7   ?strZkp?vZbdr8   ?t?gZuidZwdZgdZtkZsid?encodeZ	hexdigestZshZfwn0ZfwnZuhZhnr   r   r   r   ?<module>   sr   $

,










$,,