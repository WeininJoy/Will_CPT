import numpy as np
import matplotlib.pyplot as plt
import cplot
import numpy as np
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import cmath
import math
from scipy.special import gamma
import scipyx as spx
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr


plt.style.use('seaborn-poster')

class Universe:
    def __init__(self, Lambda, R, k_curved): # k_curved = - 1.2  # close:>0, open:<0, flat:0  , abs(value) < 1 : physical 
        
        ### cosmological constants
        self.Lambda = Lambda   # dark energy
        self.R = R    # radiation
        self.k_curved = k_curved
        self.K = k_curved * 1. / 3. * math.sqrt( 4.* Lambda * R )   

    def eta_tot(self):
        result = quad(lambda a: 1./self.a_dot(a), 0, np.inf)
        a_list = np.linspace(0, 10, 1000)
        return result[0]

    #################
    # scale factor a(eta)
    #################
    
    def a_dot(self, a): 
        return math.sqrt( 1./3. * ( self.R - 3*self.K* a**2 + self.Lambda* a**4 ) )

    def a(self, eta):
        m = 3.*self.K / (2.*self.Lambda*self.R) * (3.*self.K + cmath.sqrt( (3*self.K)**2 - 4*self.Lambda*self.R ) ) - 1
        m = m.real
        alpha = cmath.sqrt( self.R*(1+m) / (3*self.K) )
        if alpha.real < 0: # choose the positive root
            alpha = - alpha

        beta = 1./alpha * math.sqrt( self.R / 3. )
        u = beta * eta
        sn, cn, dn, ph = spx.ellipj(u, m)

        return alpha * sn
    
    def plot_a(self):

        eta_tot = self.eta_tot()
        
        ##### make plot in complex plane 
        # plt = cplot.plot(
        #     a,
        #     (-eta_tot, eta_tot, 400),
        #     (-eta_tot, eta_tot, 400)
        # )
        # plt.show()

        ##### make plot in real axis
        eta_list = np.linspace(-eta_tot, eta_tot, 1000)
        plt.plot(eta_list, [self.a(eta) for eta in eta_list])
        plt.show()

    #################
    # Phi solution
    #################
    
    def solve_Phi(self, Phi_i, Phi_dot_i, k_scale):

        def Phi_ode(eta, y):
            Phi, Phi_dot = y
            a = self.a(eta)
            Phi_2dot = -4 * self.a_dot(a)/a * Phi_dot - 1./3. * ( 4. * a**2 * self.Lambda +  k_scale**2 ) * Phi
            return [Phi_dot, Phi_2dot]
        
        eta_tot = self.eta_tot()
        y0 = [Phi_i, Phi_dot_i]
        return solve_ivp(Phi_ode, [1.e-5, eta_tot], y0, rtol=1e-10, atol=1e-10)

    def plot_Phi(self, Phi_i, Phi_dot_i, k_scale):
        
        ### Phi solution in flat universe (suppose Phi_dot_i=0.0)
        def Phi_flat(eta): 
            wolfSession = WolframLanguageSession()
            wolfSession.evaluate(wl.Needs("targetFunctions`"))
            
            a = round(self.a(eta).real, 12)
            result = Phi_i * wolfSession.evaluate(wl.targetFunctions.PhiFlat((a, k_scale)))
            wolfSession.terminate()
            return result
        
        def Phi_curved(eta):
            wolfSession = WolframLanguageSession()
            wolfSession.evaluate(wl.Needs("targetFunctions`"))
            
            a = round(self.a(eta).real, 12)
            result = Phi_i * wolfSession.evaluate(wl.targetFunctions.PhiCurved((a, k_scale, self.k_curved)))
            wolfSession.terminate()
            return result
        
        
        eta_tot = self.eta_tot()
        eta_list = np.linspace(1.e-5, eta_tot, 100)

        ## generate Phi_curved_list 
        # Phi_curved_list = []
        # for i in range(len(eta_list)):
        #     try:                      
        #         Phi_curved_list.append(Phi_curved(eta_list[i]))
        #     except:                   
        #         Phi_curved_list.append(-1)
        
        # print(Phi_curved_list)
    

        ## generate Phi_flat_list 
        # Phi_flat_list = []
        # for i in range(len(eta_list)):
        #     try:                      
        #         Phi_flat_list.append(Phi_flat(eta_list[i]))
        #     except:                   
        #         Phi_flat_list.append(-1)
        
        # print(Phi_flat_list)

        # Lambda=1., R=1., k_curved=-1.2, Phi_i=1., Phi_dot_i=0., k_scale=10, data_num=100
        Phi_curved_list = [0.5461162560960962, 0.5448059873594391, 0.5408913925289607, 0.5344150229617614, 0.5254470908657733, 0.5140844329334464, 0.5004490851157974, 0.48468649355294174, 0.46696339297019, 0.4474653895085488, 0.42639428992769307, 0.40396522323425577, 0.38040360401814616, 0.35594198907246133, 0.33081688015117944, 0.3052655260196935, 0.2795227761866081, 0.25381803708871364, 0.22837237878896022, 0.20339583682248, 0.1790849495029579, 0.15562056604590524, 0.13316595533120953, 0.11186523908844502, 0.0918421669743229, 0.0731992444581576, 0.0560172178120894, 0.04035491399582976, 0.026249426833109556, 0.013716634900000111, 0.0027520309209646925, -0.006668162576265973, -0.014585621030781983, -0.02105832246682502, -0.026158765355368742, -0.02997203831309853, -0.032593781589076075, -0.03412808083983254, -0.034685333426678955, -0.03438012642229473, -0.03332916373151707, -0.0316492772713025, -0.029455554090794978, -0.026859607725392958, -0.023968018061872928, -0.02088095963893346, -0.01769103372990883, -0.014482314838097931, -0.011329617509194067, -0.008297984698560476, -0.005442394453085484, -0.002807677431723772, -0.00042863390996232987, 0.0016696645693804856, 0.0034714067415078146, 0.004969426698739351, 0.006164464784139573, 0.007064327958326414, 0.007682974096748229, 0.008039544906333247, 0.008157371828536169, 0.008062978454110513, 0.007785101651411456, 0.0073537518559669635, 0.006799330841917159, 0.0061518228597759895, 0.005440072348531159, 0.004691158585446434, 0.003929874696788151, 0.0031783154897151323, 0.0024555756500428336, 0.001777557049887944, 0.0011568812840932919, 0.000602901160823862, 0.0001218027578716952, -0.00028321213874275754, -0.0006116748425714329, -0.0008655425237324868, -0.001048848309484293, -0.0011673224298096624, -0.001227998039398054, -0.0012388149167442055, -0.0012082334746424475, -0.0011448704464633703, -0.0010571662782534255, -0.0009530927023379495, -0.0008399072424867574, -0.0007239595556098739, -0.0006105526038726768, -0.0005038597279477711, -0.0004068968097685105, -0.0003215469223656959, -0.00024863321245106797, -0.00018803429117282986, -0.0001388351571905665, -9.950567486205874e-05, -6.809790282380994e-05, -4.2453130955124994e-05, -2.040934515385717e-05, 0.007169013844034781]
        Phi_curved_list = [Phi/Phi_curved_list[0] for Phi in Phi_curved_list]
        plt.plot(eta_list, Phi_curved_list, label = 'Phi_curved_anal')

        Phi_flat_list = [0.9999999996666666, 0.9976001587070222, 0.9904232659374467, 0.9785264137595685, 0.962004572569002, 0.940990319648467, 0.9156534398012232, 0.8862003779761556, 0.8528735186333902, 0.8159502612167272, 0.7757418560240824, 0.7325919599956531, 0.6868748677265103, 0.638993369589856, 0.5893761864351836, 0.5384749293540164, 0.4867605337221027, 0.43471912001034485, 0.3828472397045308, 0.3316464743551972, 0.28161736944675536, 0.23325270345977786, 0.18703011675114095, 0.1434041551417944, 0.10279781992638337, 0.06559375920101132, 0.03212528453015827, 0.0026674509558240354, -0.022571505109335286, -0.043458019221785056, -0.0599399411337103, -0.07205172234914374, -0.0799175814603911, -0.08375218071838848, -0.08385838148344883, -0.08062172134929438, -0.07450137778630163, -0.0660175548925988, -0.05573544898835637, -0.04424620646909778, -0.0321455668840784, -0.02001116089019951, -0.008379675066203005, 0.002274732068413587, 0.011559310795202787, 0.019177510716013185, 0.02493850649128896, 0.028760639900282436, 0.030668431987796785, 0.03078336151443304, 0.02930915499092097, 0.026512819769514744, 0.02270301077908638, 0.018207506398104766, 0.013351558315252875, 0.008438682222907331, 0.0037351063208219937, -0.0005413518819302737, -0.004227688907328784, -0.007217557048572728, -0.009459074100271758, -0.010949392356630408, -0.011726958906449198, -0.011862425605365736, -0.011449098754055967, -0.010593682547675693, -0.009407895761612526, -0.008001354371907092, -0.006475935048494815, -0.004921680352866136, -0.0034141842001747446, -0.0020133083475358487, -0.0007630257969333041, 0.000307839222077615, 0.0011842082931377879, 0.0018629028705352219, 0.00235055058262253, 0.002661408025911565, 0.0028152310293059098, 0.00283529073692865, 0.0027466038349621587, 0.002574419009392118, 0.002342979814347372, 0.002074566636767948, 0.0017888071266151985, 0.0015022348910192918, 0.0012280698929325947, 0.0009761902860703875, 0.000753263810618604, 0.0005630068674683248, 0.00040654054464498476, 0.0002828148200197335, 0.00018907461074926313, 0.00012134404898255809, 7.490816048157846e-05, 4.477388398831938e-05, 2.6095013227534184e-05, 1.4548124357403996e-05, 6.64885185075604e-06, -0.053551214689248165]
        plt.plot(eta_list, Phi_flat_list, label = 'Phi_flat_anal')

        ### Phi solution in curved universe
        Phi_sol = self.solve_Phi(Phi_i, Phi_dot_i, k_scale)
        plt.plot(Phi_sol.t, Phi_sol.y[0], label = 'Phi_curved_num')
        plt.legend()
        plt.savefig("Phi_sol.pdf")
        plt.show()

        

universe = Universe(Lambda=1., R=1., k_curved=-1.2)
# universe.plot_a()
Phi_i, Phi_dot_i, k_scale = 1., 0., 10
universe.plot_Phi(Phi_i, Phi_dot_i, k_scale)