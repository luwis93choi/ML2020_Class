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
        self.cov = []		# 각 Gaussian Cluster별 Covariance Matrix 리스트
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

        # 각 Gaussian Cluster의 평균점 (Cluster 중심 좌표)를 랜덤 좌표로 초기화하지 않는다면...
        if random_init is False:
            # 모든 Gaussian Cluster의 평균점 (Cluster 중심 좌표)를 (0, 0)으로 초기화함
            self.mu = np.zeros((self.number_of_sources, 2))	

        # 각 Gaussian Cluster의 평균점 (Cluster 중심 좌표)를 랜덤 좌표로 초기화하면
        else:
            # 각 Gaussain Cluster의 평균점 (Cluster 중심 좌표)를 데이터셋 범위내의 랜덤 좌표로 할당함
            self.mu = np.random.randint(min(self.X[:,0]),max(self.X[:,0]),size=(self.number_of_sources,self.X.shape[1]))	

        # 각 Gaussian Cluster의 Covariance Matrix를 Identity Matrix로 초기화함
        for i in range(self.number_of_sources):
            self.cov.append(np.identity(self.X.shape[1]))

        # 각 Gaussian Cluster가 데이터셋에 대해 동일한 비율만큼 차지하는 것으로 초기화함
        self.pi = np.ones(self.number_of_sources) / self.number_of_sources

        log_likelihoods = []	# 데이터셋에 대한 Log-Likelihood를 저장하는 리스트

        # 정해진 반복횟수만큼 EM 알고리즘을 통해 각 Gaussian Cluster의 중심(Mean)에 최대한 많은 데이터가 모일 수 있도록 중심점(Mean)과 포함 범위(Covariance Matrix)를 변경해나감
        for i in range(self.iterations):

            ######################################################################
            ### E-Step : 데이터셋의 각 데이터가 특정 Cluster에 해당되는 확률을 구함 ###
            ######################################################################
            r_ic = np.zeros((self.X.shape[0], self.number_of_sources))
            # 데이터셋(self.X.shape[0] : 데이터셋 개수)이 각 Gaussian Cluster(self.number_of_sources : Cluster 개수)에 속할 확률을 리스트로 저장함 

            # 각 Gaussain Cluster의 평균, Covariance Matrix, Pi (데이터별 소속될 비율)에 대해 전체 데이터셋의 소속 정도를 평가함
            for m, co, p, r in zip(self.mu, self.cov, self.pi, range(len(r_ic[0]))):

                co += self.reg_cov	# Singularity Issue를 방지하기 위해 Covariance Matrix가 0이 되지 않게 매우 작은 값을 더함

                mn = multivariate_normal(mean=m, cov=co)	# 현재 사용하는 평균, Covariance를 반영한 Gaussian Cluster 준비

                # 현재 Gaussian Cluster에 대해 모든 데이터셋이 소속될 확률을 구함
                # 해당 확률을 전체 Cluster에 대한 소속 확률의 합으로 0 ~ 1사이로 Noramlize함
                r_ic[:, r] = p * mn.pdf(self.X) / np.sum([pi_c * multivariate_normal(mean=mu_c, cov=cov_c).pdf(self.X) for pi_c, mu_c, cov_c in zip(self.pi, self.mu, self.cov+self.reg_cov)], axis=0)

            #############################################################################################################
            ### M-Step : 각 데이터의 Cluster별 소속 비율을 반영해서 Gaussian Cluster의 평균(중심점 좌표)를 더 많은 소속 비율을 가진 ###
            ###          데이터를 향해서 변동이 되도록 평균을 Weighted Mean, Covriance를 Weighted Covariance로 변경함           ###
            #############################################################################################################
            self.mu = []		# 각 Gaussian Cluster별 평균을 새롭게 업데이트하기 위해 준비함
            self.cov = []		# 각 Gaussian Cluster별 Covariance Matrix를 새롭게 업데이트하기 위해 준비함
            self.pi = []		# 각 Gaussian Cluster별 데이터셋에 대한 점유율 업데이트하기 위해 준비함
            log_likelihood = []		# 매 Iteration마다 Log-Likelihood를 저장하기 위한 리스트
            
            # 각 Gaussian Cluster별 평균(중심점 좌표), Covariance Matrix, 점유율을 새롭게 업데이트함
            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:, c], axis=0)	# 각 Gaussian Cluster에 대해 전체 데이터셋의 소속 비율합 업데이트
                # 만약 소속 비율합이 0이 되는 경우 Gaussian Cluster 평균 연산시 발생하는 문제를 방지하기 위해 0이 되는 경우 1로 만듬
                if (m_c == 0.0): m_c = 1
                mu_c = (1/m_c) * np.sum(self.X * r_ic[:, c].reshape(len(self.X), 1), axis=0)	# 각 Gaussian Cluster의 평균(중심점 좌표)를 연산함
                self.mu.append(mu_c)	# 각 Gaussian Cluster의 평균(중심점 좌표)를 업데이트함 (Gaussian Cluster 중심에 더 많은 데이터가 소속되는 방향으로 중심점 좌표가 업데이트됨)

                # 각 Gaussian Cluster의 Covariance를 연산하고 업데이트함
                self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c)))+self.reg_cov)	

                self.pi.append(m_c/np.sum(r_ic))	# 각 Gaussian Cluster의 데이터셋에 대한 점유율 업데이트

            # 전체 데이터셋의 각 Gaussian Cluster에 해당되는 값들의 합을 Log-Likelihood로 변환하여 저장함
            log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i],self.cov[j]).pdf(self.X) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))

            ####################################################
            ### Singularity Check & Random Re-Initialization ###
            ####################################################

            ### Singularity : 데이터 1개가 Gaussian Cluster 평균/중심점에 완전 일치하게 되면서 해당 지점을 중심으로 Gaussian Cluster가 Overfitting되는 현상
            ###             : 데이터 1개 중심으로 Gaussian Cluster가 Overfitting되면 해당 지점을 중심으로 Gaussian이 매우 크게 나타나기 때문에 Covariance가 매우 낮게 나타남
            ###             : 이러한 현상은 Covariance Matrix가 Singular Matrix (Determinant = 0)인 경우 발생하며, 
            ###               이 때 Gaussian Cluster의 중심점을 랜덤하게 다른 위치로 옮겨서 다르게 Cluster를 구성하게 만듬

            singularity = np.zeros(self.number_of_sources)	# 각 Gaussian Cluster별 Singularity 존재여부를 저장하기 위한 리스트

            # 각 Gaussian Cluster의 업데이트된 Covariance Matrix의 Determinant가 특정값 이하인 경우 해당 Cluster에 Singularity가 발생했다고 간주함
            # 컴퓨터 연산을 이용한 Determinant 연산은 0을 0이 아닌 매우 낮은 값으로 산출할 수 있기 때문에, 기준을 0이 아닌 특정 매우 낮은값 이하로 결정함
            for co, cluster_idx in zip(self.cov, range(self.number_of_sources)):
                if abs(np.linalg.det(co)) < 1e-2:	# Determinant의 크기가 0.01 이하인 경우 Singularity라 간주함
                    singularity[cluster_idx] = 1

            print('--- Singluarity Check : {} ---'.format(singularity))

            # 각 Gaussian Cluster 중 Singularity가 발생하면 평균/중심점으로 다시 랜덤하게 배정하고, Covariance Matrix를 초기화함
            # 그리고 해당 Gaussian Cluster의 점유 비율을 상대적을 높게 배정해서 기존의 Gaussian Cluster 사이에 침투할 수 있도록 허용해줌
            for singularity_idx in range(len(singularity)):
                if singularity[singularity_idx] == 1:
                    self.mu[singularity_idx] = np.random.randint(min(self.X[:,0]), max(self.X[:,0]), size=(self.X.shape[1]))	# Gaussian Cluster 중심점을 랜덤하게 재할당

                    self.cov[singularity_idx] = np.identity(self.X.shape[1]) + self.reg_cov     # Gaussian Cluster의 Covariance Matrix 재초기화
                    
                    self.pi[singularity_idx] = 1	# Gaussian Cluster의 점유 비율을 상대적으로 높게 배정 / 기존 Cluster 영역을 침투할 수 있게 허용함

            ###############################
            ### Clustering 결과 Plotting ###
            ###############################

            # 매 Iteration마다 Gaussian Cluster의 평균/중심점을 기준으로 등고선을 그림
            plt.subplot(1, 2, 1)
            plt.title('EM Algorithm at Iteration {}'.format(i))
            plt.xlim(1.5 * min(self.X[:,0]), 1.5 * max(self.X[:,0]))
            plt.ylim(1.5 * min(self.X[:,1]), 1.5 * max(self.X[:,1]))
            plt.grid()
            x, y = np.meshgrid(np.sort(self.X[:, 0]), np.sort(self.X[:, 1]))
            XY = np.array([x.flatten(), y.flatten()]).T
            plt.scatter(self.X[:,0],self.X[:,1])
            for m,c in zip(self.mu,self.cov):
                c += self.reg_cov
                multi_normal = multivariate_normal(mean=m,cov=c)
                plt.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
                plt.scatter(m[0],m[1],c='grey',zorder=10,s=100)

            # 현재까지 수행한 Iteration에서 산출된 Log-Likelihood를 그래프로 그림
            plt.subplot(1, 2, 2)
            plt.title('Log-Likelihood')
            plt.grid()
            plt.plot(range(len(log_likelihoods)),log_likelihoods)

            # 지속적으로 GMM Clustering 결과를 그림
            plt.pause(0.005)
            plt.show(block=False)
            plt.clf()

# 메인 함수 실행
if __name__ == '__main__':

    # Blob 또는 Moon 데이터 500개 데이터셋에 대해 200회 EM Algorithm을 통해 GMM Clustering 수행함
    GMM = GMM(iterations=200, dataset_type='blob', n_samples=500)	
    GMM.run(random_init=True)
