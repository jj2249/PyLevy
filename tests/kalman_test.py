from PyLevy.filtering.filters import KalmanFilterMarcos
import numpy as np
import matplotlib.pyplot as plt


# https://machinelearningspace.com/object-tracking-python/
def test_LGSSM_KF():
    """
    x(k) = Ax(k - 1) + Uu(k - 1) + w(k), w(k) = N(0, Q)
    y(k) = Hx(k) + v(k), v(k) = N(0, R)
    """
    A = np.eye(2)
    B = np.eye(2)
    Q = 1 ** 2 * np.eye(2)
    H = np.array([1, 0]).reshape((1, 2))
    R = 1 ** 2
    time_ax = np.arange(0, 100, 1)
    """ Initialise filter """
    L = np.linalg.cholesky(Q)
    x = L@np.random.randn(Q.shape[0], 1)
    mean_update = np.zeros(shape=(1, 2))
    cov_update = np.eye(2)
    position_preds = [mean_update[0,0]]
    observations = [0]
    states = [x]
    true_position = [x[0]]
    kf = KalmanFilterMarcos(mean_update, cov_update, B=B, H=H)

    """ Run Kalman Filter"""
    for i in range(1, len(time_ax)):
        x = A @ states[i - 1] + L@np.random.randn(Q.shape[0], 1)
        y = H @ x + np.random.normal(loc=0., scale=R)
        states.append(x)
        observations.append(y)
        true_position.append(x[0])

        pred_mean, pred_cov = kf.predict_given_jumps(A=A, full_noise_covar=Q)
        position_preds.append(pred_mean[0][0])
        kf.correct(y, obs_noise=1e-5)

    print()
    plt.plot(time_ax, position_preds, label="Kalman Estimate", linestyle='dashed', color='r')
    plt.plot(time_ax, true_position, label="State", color='b')
    # plt.plot(time_ax, observations, label="Observations", color='black')
    plt.title("Kalman Filter for LGSSM")
    plt.legend()
    plt.xlabel("Time /s")
    plt.ylabel("Position /m")
    plt.show()

test_LGSSM_KF()