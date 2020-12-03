import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import numpy as np
from scipy.stats import multivariate_normal

class GMM:

    # GMM 생성자
    def __init__(self, iterations, dataset_type='blob', n_samples=2000):

        self.iterations = iterations	# GMM을 최적화기 위한 EM Algorithm 반복 횟수
        self.mu = None		# 각 Gaussian Cluster별 Mean값 리스트
        self.cov = None		# 각 Gaussian Cluster별 Covariance Matrix 리스트
        self.pi = None		# 데이터셋의 각 Gaussian Cluster에 대한 소속 비율

	# 데이터셋 종류에 따라 준비함
        if dataset_type is 'blob':	# Blob 데이터셋 준비

            X, Y = make_blobs(cluster_std=1.5, random_state=20, n_samples=n_samples, centers=5)		# Blob 데이터셋 생성
            X = np.dot(X, np.random.RandomState(0).randn(2, 2))		# Blob 데이터셋이 타원형이 될 수 있게 만듬	
            self.X = X	# 데이터셋 멤버 변수화
            self.number_of_sources = len(np.unique(Y))	# 데이터셋의 클러스터 종류 개수 

        elif dataset_type is 'moon':	# Moon 데이터셋 준비
        
            X, Y = make_moons(n_samples=n_samples, noise=0.05, random_state=0)	# Moon 데이터셋 생성
            self.X = X	# 데이터셋 멤버 변수화
            self.number_of_sources = len(np.unique(Y))	# 데이터셋의 클러스터 종류 개수

    # EM Algorithm을 이용한 GMM Clustering 수행
    def run(self, random_init=True):
        
        self.reg_cov = 1e-6 * np.identity(self.X.shape[1])	# Singularity Issue (Gaussian Cluster가 한 점에 대해 Overfitting 되는 현상)를 방지하기 위해 
								# Covariance Matrix가 0 Matrix가 되지 않도록 매우 작은 값을 가지는 Identity Matrix를 더해줌
								
								# GMM Singularity Issue 해결방법 1 : 매우 작은 값을 추가한 Identity Matrix를 Covariance Matrix에 더함 (sklearn 구현방식)
								# GMM Singularity Issue 해결방법 2 : 특정 Threshold 기준으로 Singularity 발생 여부를 확인되면 Gaussian Cluster의 Mean 위치를 다른 랜덤한 위치로 재설정함
								# 출처 : https://stats.stackexchange.com/questions/219302/singularity-issues-in-gaussian-mixture-model
        
        if random_init is False:
            self.mu = np.zeros((self.number_of_sources, 2))
        
        else:
            self.mu = np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.number_of_sources,self.X.shape[1]))

        self.cov = np.zeros((self.number_of_sources, self.X.shape[1], self.X.shape[1]))
        
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim], 1)

        self.pi = np.ones(self.number_of_sources) / self.number_of_sources

        log_likelihoods = []

        for i in range(self.iterations):

            ### E-Step ###
            r_ic = np.zeros((len(self.X), len(self.cov)))

            for m, co, p, r in zip(self.mu, self.cov, self.pi, range(len(r_ic[0]))):

                co += self.reg_cov
                
                mn = multivariate_normal(mean=m, cov=co)

                r_ic[:, r] = p * mn.pdf(self.X) / np.sum([pi_c * multivariate_normal(mean=mu_c, cov=cov_c).pdf(self.X) for pi_c, mu_c, cov_c in zip(self.pi, self.mu, self.cov+self.reg_cov)], axis=0)

            ### M-Step ###
            self.mu = []
            self.cov = []
            self.pi = []
            log_likelihood = []
            
            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:, c], axis=0)
                if (m_c == 0.0):
                    m_c = 1
                mu_c = (1/m_c) * np.sum(self.X * r_ic[:, c].reshape(len(self.X), 1), axis=0)
                self.mu.append(mu_c)

                self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c)))+self.reg_cov)

                self.pi.append(m_c/np.sum(r_ic))

            log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i],self.cov[j]).pdf(self.X) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))

            ### Singularity Check & Random Re-Initialization ###
            singularity = np.zeros(self.number_of_sources)
        
            for co, cluster_idx in zip(self.cov, range(self.number_of_sources)):
                if abs(np.linalg.det(co)) < 1e-2:
                    singularity[cluster_idx] = 1

            print('--- Singluarity Check : {} ---'.format(singularity))

            for singularity_idx in range(len(singularity)):
                if singularity[singularity_idx] == 1:
                    self.mu[singularity_idx] = np.random.randint(min(self.X[:,0]), max(self.X[:,0]), size=(self.X.shape[1]))
                    
                    self.cov[singularity_idx] = np.identity(self.X.shape[1])
                    self.cov[singularity_idx] += self.reg_cov
                    
                    self.pi[singularity_idx] = 1
            #####################################################

            plt.subplot(1, 2, 1)
            plt.title('EM Algorithm at Iteration {}'.format(i))
            plt.xlim(1.5 * min(self.X[:,0]), 1.5 * max(self.X[:,0]))
            plt.ylim(1.5 * min(self.X[:,1]), 1.5 * max(self.X[:,1]))
            x, y = np.meshgrid(np.sort(self.X[:, 0]), np.sort(self.X[:, 1]))
            XY = np.array([x.flatten(), y.flatten()]).T
            plt.scatter(self.X[:,0],self.X[:,1])
            for m,c in zip(self.mu,self.cov):
                c += self.reg_cov
                multi_normal = multivariate_normal(mean=m,cov=c)
                plt.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
                plt.scatter(m[0],m[1],c='grey',zorder=10,s=100)

            plt.subplot(1, 2, 2)
            plt.title('Log-Likelihood')
            plt.plot(range(len(log_likelihoods)),log_likelihoods)

            plt.pause(0.005)
            plt.show(block=False)
            plt.clf()

GMM = GMM(200, dataset_type='blob', n_samples=500)
GMM.run(random_init=True)
