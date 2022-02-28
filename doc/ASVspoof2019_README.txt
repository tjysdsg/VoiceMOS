=====================================================================================================
ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database
=====================================================================================================

Names (equal contribution): 
Junichi Yamagishi (1)(2)
Massimiliano Todisco (3)
Md Sahidullah (4)
Héctor Delgado (3)
Xin Wang (1)
Nicholas Evans (3)
Tomi Kinnunen (5)
Kong Aik Lee (6)
Ville Vestman (5)
Andreas Nautsch (3)

Affiliations: 
(1) National Institute of Informatics, Japan
(2) University of Edinburgh, UK
(3) EURECOM, France
(4) Inria, France 
(5) University of Eastern Finland, Finland
(6) NEC, Japan 

Copyright (c) 2019  
The Centre for Speech Technology Research (CSTR)
University of Edinburgh

―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

REFERENCE
When publishing studies or results on the ASVspoof2019 corpus, please cite:

@InProceedings{Todisco2019,
  Title                    = {{ASV}spoof 2019: {F}uture {H}orizons in {S}poofed and {F}ake {A}udio {D}etection},
  Author                   = {Todisco, Massimiliano and Wang, Xin and Sahidullah, Md and Delgado, H ́ector and Nautsch, Andreas and Yamagishi, Junichi and Evans, Nicholas and Kinnunen, Tomi and Lee, Kong Aik},
  booktitle		   = {Proc. of Interspeech 2019},
  Year                     = {2019}
}

―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

DIRECTORY STRUCTURE

  ./ASVspoof2019_root
  	  	--> LA  
          		--> ASVspoof2019_LA_asv_protocols
          		--> ASVspoof2019_LA_asv_scores
	  		--> ASVspoof2019_LA_cm_protocols
          		--> ASVspoof2019_LA_dev
          		--> ASVspoof2019_LA_eval
	  		--> ASVspoof2019_LA_train
	  		--> README.LA.txt
  	  	--> PA 
          		--> ASVspoof2019_PA_asv_protocols
          		--> ASVspoof2019_PA_asv_scores
	  		--> ASVspoof2019_PA_cm_protocols
          		--> ASVspoof2019_PA_dev
          		--> ASVspoof2019_PA_eval
	  		--> ASVspoof2019_PA_train
	  		--> README.PA.txt
	  	--> asvspoof2019_evaluation_plan.pdf
	  	--> asvspoof2019_Interspeech2019_submission.pdf
	  	--> README.txt

―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

OVERVIEW
This is a database used for the Third Automatic Speaker Verification Spoofing and Countermeasuers Challenge, for short, ASVspoof 2019 (http://www.asvspoof.org) organized by Junichi Yamagishi, Massimiliano Todisco, Md Sahidullah, Héctor Delgado, Xin Wang, Nicholas Evans, Tomi Kinnunen, Kong Aik Lee, Ville Vestman, and Andreas Nautsch in 2019.

The ASVspoof challenge aims to encourage further progress through (i) the collection and distribution of a standard dataset with varying spoofing attacks implemented with multiple, diverse algorithms and (ii) a series of competitive evaluations for automatic speaker verification. 

The ASVspoof 2019 challenge follows on from three special sessions on spoofing and countermeasures for automatic speaker verification held during INTERSPEECH 2013 [1], 2015 [2], and 2017 [3]. While the first edition in 2013 was targeted mainly at increasing awareness of the spoofing problem, the 2015 edition included the first challenge on the topic, accompanied by commonly defined evaluation data, metrics and protocols. The task in ASVspoof 2015 was to design countermeasure solutions capable of discriminating between bona fide (genuine) speech and spoofed speech produced using either text-to-speech (TTS) or voice conversion (VC) systems. The ASVspoof 2017 challenge focused on the design of countermeasures aimed at detecting replay spoofing attacks that could, in principle, be implemented by anyone using common consumer-grade devices.

The ASVspoof 2019 challenge extends the previous challenge in several directions. The 2019 edition is the first to focus on countermeasures for all three major attack types, namely those stemming from TTS, VC and replay spoofing attacks. Advances with regards to the 2015 edition include the addition of up-to-date TTS and VC systems that draw upon substantial progress made in both fields during the last four years. ASVspoof 2019 thus aims to determine whether the advances in TTS and VC technology post a greater threat to automatic speaker verification and the reliability of spoofing countermeasures.

Advances with regards to the 2017 edition concern the use of a far more controlled evaluation setup for the assessment of replay spoofing countermeasures. Whereas the 2017 challenge was created from the recordings of real replayed spoofing attacks, the use of an uncontrolled setup made results somewhat difficult to analyse. A controlled setup, in the form of replay attacks simulated using a range of real replay devices and carefully controlled acoustic conditions is adopted in ASVspoof 2019 with the aim of bringing new insights into the replay spoofing problem.

Last but not least, the 2019 edition aligns ASVspoof more closely with the field of automatic speaker verification. Whereas the 2015 and 2017 editions focused on the development and assessment of stand-alone countermeasures, ASVspoof 2019 adopts for the first time a new ASV-centric metric in the form of the tandem decision cost function (t-DCF) [4].

The ASVspoof 2019 database encompasses two partitions for the assessment of logical access (LA) and physical access (PA) scenarios. Both are derived from the VCTK base corpus [5] which includes speech data captured from 107 speakers (46 males, 61 females). Both LA and PA databases are themselves partitioned into three datasets, namely training, development and evaluation which comprise the speech from 20 (8 male, 12 female), 10 (4 male, 6 female) and 48 (21 male, 27 female) speakers respectively. The three partitions are disjoint in terms of speakers, and the recording conditions for all source data are identical. While the training and development sets contain spoofing attacks generated with the same algorithms/conditions (designated as known attacks), the evaluation set also contains attacks generated with different algorithms/conditions (designated as unknown attacks). Reliable spoofing detection performance therefore calls for systems that generalise well to previously-unseen spoofing attacks. 

Full descriptions are available in the ASVspoof 2019 evaluation plan [6].

Below are some details about the database:

1. Training and development data for the LA scenario are included in 'ASVspoof2019_LA_train'  ' ASVspoof2019_LA_dev'. Training dataset contains audio files with known ground-truth, which can be used to train systems to distinguish genuine and spoofed speech. The development dataset contains audio files with known ground-truth which can be used for the development of spoofing detection algorithms. Likewise, training and development data for the PA scenario are included in 'ASVspoof2019_PA_train'  ' ASVspoof2019_PA_dev'. 

2. Evaluation data for LA and PA are available in 'ASVspoof2019_LA_eval'  and 'ASVspoof2019_PA_eval', respectively.

3. Dev and eval enrollment data for ASV are available in 'ASVspoof2019_{LA,PA}_dev' and 'ASVspoof2019_{LA,PA}_eval', respectively.

4. Protocol and keys are available in 'ASVspoof2019_LA_{cm,asv}_protocols'  and ASVspoof2019_PA_{cm,asv}_protocols, respectively.

5. Additional README.LA.txt files and README.pA.txt are included in packages. They are the extended version of ASVspoof2019_{LA,PA}_instructions_v1.txt originally used for the challenge to explain the database.

6. About how to compute the EERs and t-DCF, please refer the evaluation plan included in this repository.

7. To compare with the challenge results, please check the summary paper of the challenge included in this repository.

8. The baseline results based on LFCC and CQCC can be reproduced using publicly released Matlab-based implementation of a replay attack spoofing detector http://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip

COPYING 
You are free to use this database under Open Data Commons Attribution License (ODC-By). 

Regarding Open Data Commons Attribution License (ODC-By), please see 
https://opendatacommons.org/licenses/by/1.0/index.html

THIS DATABASE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS DATABASE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


ACKNOWLEDGEMENTS
The new ASVspoof 2019 database is the result of more than six months of intensive work including contributions from leading academic and industrial research laboratories. The ASVspoof consortium would like to strongly thank the following 14 organizations and 27 persons who contributed to this database:

* Ms. Avashna Govender and Dr. Srikanth Ronanki from University of Edinburgh, UK, 
* Prof. Tomoki Toda and Mr. Yi-Chiao Wu from Nagoya University, Japan, Japan, Mr. Wen-Chin Huang, Mr. Yu-Huai Peng, and Dr. Hsin-Te Hwang from Academia Sinica, Taiwan. 
* Prof. Zhen-Hua Ling and Mr. Jing-Xuan Zhang from University of Science and Technology of China, China, and Mr. Yuan Jiang and Ms. Li-Juan Liu from iFlytek Research, China.
* Dr. Ingmar Steiner from Saarland University and DFKI GmbH, Germany and Dr. Sébastien Le Maguer from Adapt centre, Trinity College Dublin, Ireland
* Dr. Kou Tanaka and Dr. Hirokazu Kameoka from NTT Communication Science Laboratories, Japan
* Mr. Kai Onuma, Mr. Koji Mushika, and Mr. Takashi Kaneda from HOYA, Japan  
* Dr. Markus Becker, Mr. Fergus Henderson, Dr. Rob Clark from the Google Text-To-Speech team, Google LLC and Dr. Yu Zhang, Dr. Quan Wang from the Google Brain team, and Deepmind, Google LLC
* Prof. Driss Matrouf and Prof. Jean-François Bonastre from LIA, University of Avignon, France
* Mr. Lauri Juvela and Prof. Paavo Alku from Aalto University, Finland

This work was partially supported by JST CREST Grant Number JPMJCR18A6 (VoicePersonae project), Japan and by MEXT KAKENHI Grant Numbers (16H06302, 16K16096, 17H04687, 18H04120, 18H04112, 18KT0051), Japan; The work is also partially supported by research funds received from the French Agence Nationale de la Recherche (ANR) in connection with the bilateral VoicePersonae (with JST CREST in Japan) and RESPECT (with DFG in Germany) collaborative research projects. The is also supported by Academy of Finland (proj. no. 309629 entitled ``NOTCH: NOn-cooperaTive speaker CHaracterization''). The authors at the University of Eastern Finland also gratefully acknowledge the use of computational infrastructures at CSC -- IT Center for Science, and support of NVIDIA Corporation with the donation of the Titan V GPU used in this research. The work is also partially supported by Region Grand Est, France.

REFERENCES
[1] Nicholas Evans, Tomi Kinnunen and Junichi Yamagishi, "Spoofing and countermeasures for automatic speaker verification", Interspeech 2013,  925-929, August 2013

[2] Zhizheng Wu, Tomi Kinnunen, Nicholas Evans, Junichi Yamagishi, Cemal Hanilc, Md Sahidullah Aleksandr Sizov, "ASVspoof 2015: the First Automatic Speaker Verification Spoofing and Countermeasures Challenge", Proc. Interspeech 2015  2037-2041 September 2015

[3] Tomi Kinnunen, Md Sahidullah, Héctor Delgado, Massimiliano Todisco, Nicholas Evans, Junichi Yamagishi, Kong Aik Lee, "The ASVspoof 2017 Challenge: Assessing the Limits of Replay Spoofing Attack Detection", Proc. Interspeech 2017, August 2017

[4] T. Kinnunen, K. Lee, H. Delgado, N. Evans, M. Todisco, M. Sahidullah, J. Yamagishi, and D. A. Reynolds, “t-DCF: a detection cost function for the tandem assessment of spoofing countermeasures and automatic speaker verification,” in Proc. Odyssey, June 2018.

[5] VCTK corpus: http://dx.doi.org/10.7488/ds/1994

[6] ASVSpoof evaluation plan: http://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf


POINTERS 
ASVSpoof challenge: http://www.asvspoof.org


