# path of processed news data
data_load_path = '/data/ent2ent/han/incremental/large_model_input_gs1.pkl'

# word category index
subject_ix = 1
predicate_ix = 2
object_ix = 3
noun_ix = 4
verb_ix = 5
adjective_ix = 6
adverb_ix = 7
all_ix = 8

# start and end year of data
year_base = 16
year_now = 18

# word embedding dimensionality
d_word = 300

# entity layer dimensionality
d_char = 50

# metadata layer dimensionality
d_book = 10

# mixing layer dimensionality
d_mix = 300

# impactful noun hidden state dimensionality
d_noun_hidden = d_word + (year_now - year_base + 1) * 12

# number of descriptors
num_descs = 30

# number of word choice for each descriptor
num_descs_choice = 5

# number of impactful noun choice overall
num_nouns_choice = 10

# number of negative training examples
num_negs = 15

# number of drop rate for input word layer
p_drop = 0.5

# training epochs and learning rate
n_epochs = 15
lr = 1e-3

# descriptor matrix orthogonal penalty weight
eps = 1e-1

# batch size
batch_size = 256

# number of samples selected for high scored documents
sample_sel_num = 5

# color palette
color_choices = ['blue', 'red', 'orange', 'green', 'grey', 'purple']

# number of top descritors shown in each temporal trend figure
num_top_descs = 3

# pairs that are interesting
interest_pairs = ['U.S. AND Russia', 'U.S. AND China', 'U.S. AND Syria', 'U.S. AND U.K.', 'China AND India', 'U.S. AND Canada',
                  'U.S. AND India', 'U.S. AND Japan']

from collections import defaultdict
key_event_dict = defaultdict(dict)

key_event_dict['Internation']['U.S. AND U.K.'] = {'17-01': "Therasa May visited U.S. (https://www.cnn.com/2017/01/26/politics/donald-trump-theresa-may-white-house-visit/index.html)",
                                                  '16-06': "U.K. Brexit vote (https://www.bbc.com/news/36622711)",
                                                  '17-12': "Trump set a controversial visit to U.K. (https://www.csmonitor.com/World/Europe/2017/1207/In-Trump-era-US-UK-special-relationship-faces-and-causes-new-trials)"}


key_event_dict['Internation']['U.S. AND Syria'] = {'16-10': "U.S. suspended Syria ceasefire talk (https://www.reuters.com/article/us-mideast-crisis-usa-russia-idUSKCN1231X3)",
                                                   '17-04': "Khan Shaykhun chemical attack and Shayrat missile strike (https://en.wikipedia.org/wiki/2017_Shayrat_missile_strike)",
                                                   '17-10': "ISIS 'capital' captured (https://www.nytimes.com/2017/10/17/world/middleeast/isis-syria-raqqa.html)",
                                                   '17-11': "ISIS's defeat and aftermath (https://news.antiwar.com/2017/11/17/pentagon-isis-defeated-but-us-will-stay-in-syria/)",
                                                   '18-02': "Battle of Khasham (https://en.wikipedia.org/wiki/Battle_of_Khasham)"}


key_event_dict['Internation']['U.S. AND Canada'] = {'17-01': "Trump said Nafta renegotiation to be started (https://www.bbc.com/news/world-us-canada-38713227)",
                                                    '17-04': "Trump imposed tariff on Canadian lumber (https://www.nytimes.com/2017/04/24/us/politics/lumber-tariff-canada-trump.html)",
                                                    '18-06': "Canada fought back with retaliatory tariff on U.S. products (https://www.cnbc.com/2018/06/29/canada-makes-retaliatory-tariffs-official-we-will-not-back-down.html)"}


key_event_dict['Internation']['U.S. AND Russia'] = {'17-04': "Syria airstrike (https://www.theguardian.com/world/2017/apr/07/us-airstrikes-syria-russian-american-relations-vladimir-putin)",
                                                    '17-11': "Trump and Putin's meeting at APEC (https://www.cnn.com/2017/11/09/politics/donald-trump-vladimir-putin-vietnam/index.html)",
                                                    '16-10': "U.S. officially accused Russia's hacking (https://www.theguardian.com/technology/2016/oct/07/us-russia-dnc-hack-interfering-presidential-election)",
                                                    '17-07': "Trump and Putin's first meeting (https://www.theatlantic.com/news/archive/2017/07/trump-putin/532899/)",
                                                    '18-02': "Dozens of Russians killed by U.S.-backed Syria attack (https://www.nytimes.com/2018/02/13/world/europe/russia-syria-dead.html)"}


key_event_dict['Internation']['U.S. AND China'] = {'16-12': "Trump made phone call to Taiwan's leader (https://en.wikipedia.org/wiki/Trump–Tsai_call)",
                                                   '17-08': "Section 301 investigations on China (http://www.businessinsider.com/us-begins-section-301-investigation-2017-8)",
                                                   '18-03': "Trump started issuing a series of tariffs (https://www.nytimes.com/2018/03/22/us/politics/trump-will-hit-china-with-trade-measures-as-white-house-exempts-allies-from-tariffs.html)",
                                                   '17-11': "Trump visited China (https://www.nytimes.com/2017/11/07/business/trump-china-trade.html)",
                                                   '17-04': "Xi visited U.S. (https://www.theguardian.com/us-news/2017/apr/06/trump-china-meeting-xi-jinping-mar-a-lago)"}


key_event_dict['Internation']['China AND India'] = {'16-04': "Minister of Foreign Affairs meeting (https://thediplomat.com/2016/04/foreign-ministers-of-russia-india-china-meet-in-moscow/)",
                                                    '16-11': "China and India's joint military drill (https://thediplomat.com/2016/11/china-india-hold-joint-military-drill/)",
                                                    '17-05': "India refused to attend Belt and Road Summit (https://www.thehindu.com/news/international/india-unlikely-to-participate-in-chinas-belt-and-road-forum/article18445908.ece)",
                                                    '17-06': "Doklam border standoff started (https://en.wikipedia.org/wiki/2017_China–India_border_standoff)",
                                                    '17-08': "Doklam border standoff ended (https://en.wikipedia.org/wiki/2017_China–India_border_standoff)",
                                                    '17-02': "India to develop a new missile (https://www.armyrecognition.com/weapons_defence_industry_military_technology_uk/india_to_develop_new_variant_of_brahmos_missile.html)",
                                                    '17-11': "China and India's WMCC meeting (http://mea.gov.in/press-releases.htm?dtl/29122/IndiaChina_WMCC_Meeting_November_17_2017)",
                                                    '18-03': "China and India to boost trade (http://www.xinhuanet.com/english/2018-03/27/c_137068871.htm)"}


key_event_dict['Internation']['U.S. AND India'] = {'16-06': "Modi visited U.S. and met Obama (https://www.c-span.org/video/?410278-2/president-obama-meets-prime-minister-modi-india)",
                                                  '16-12': "Trump made a complimentary phone call to Pakistan (https://www.bbc.com/news/world-asia-38165878)",
                                                  '17-06': "Modi visited U.S. and met Trump (https://www.c-span.org/video/?430524-4/us-india-relations)",
                                                  '17-10': "U.S. Secretary Of State Tillerson visited India (https://www.npr.org/sections/parallels/2017/10/26/560224471/tillerson-visit-highlights-indias-evolving-relationship-with-u-s)",
                                                  '18-06': "India imposed retaliatory tariffs on U.S. (https://www.washingtonpost.com/world/india-imposes-retaliatory-tariffs-on-us-as-global-trade-war-widens/2018/06/21/7c3a016b-1de0-497a-9635-a522bc55810a_story.html?noredirect=on&utm_term=.6ba2cd6411ed)"}


key_event_dict['Internation']['U.S. AND Japan'] = {'16-05': "Obama gave memorial speech at Hiroshima with Japanese PM Abe (https://www.nytimes.com/2016/05/28/world/asia/text-of-president-obamas-speech-in-hiroshima-japan.html)",
                                                   '16-12': "Abe visited Pearl Harbor (https://www.nytimes.com/2016/12/05/world/asia/shinzo-abe-pearl-harbor-japan.html)",
                                                   '17-02': "Abe visited Washington and met Trump (https://www.washingtonpost.com/news/monkey-cage/wp/2017/02/13/did-trump-and-abe-just-launch-a-new-chapter-in-u-s-japan-relations/?utm_term=.f4e2a17f96b6)",
                                                   '17-11': "Trump visited Japan and met Abe (https://www.whitehouse.gov/briefings-statements/president-donald-j-trumps-summit-meeting-prime-minister-shinzo-abe-japan/)",
                                                   '18-03': "Trump accepted North Korea's invitation for direct nuclear talks (https://www.cnbc.com/2018/03/29/japan-wants-its-own-bilateral-summit-with-north-korea.html)"}