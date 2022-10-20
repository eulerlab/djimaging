import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_rf_summary(w, quality_dict, model_summary, title=""):
    """Plot tRF and sRFs and quality test"""

    n_rows = 5
    fig, axs = plt.subplots(n_rows, 4, figsize=(8, n_rows * 1.9), gridspec_kw=dict(height_ratios=(1.0, 1.5, 1, 1, 1)))

    axbigs = []
    for i in np.arange(2, n_rows):
        for ax in axs[i, :].flat:
            ax.remove()
        axbig = fig.add_subplot(axs[0, 0].get_gridspec()[i, :])
        axbigs.append(axbig)
    axs = np.append(axs.flatten(), np.array(axbigs))

    if w.ndim == 3:
        sRF, sRF_max, sRF_min, tRF, peak_idxs = rf_utils.get_spatial_and_temporal_filters_3d(w)
        vmax = np.max([np.max(np.abs(w)), np.max(np.abs(w))])
    elif w.ndim == 1:
        tRF, tRF_max = w.flatten(), w.flatten()
        sRF, sRF_max = np.zeros((1, 1)), np.zeros((1, 1))
        vmax = 1
        peak_idxs = np.array([])
    else:
        raise NotImplementedError(w.shape)

    assert tRF.size == w.shape[0]

    axs[0].set(title='tRF')
    t_tRF = np.arange(-tRF.size + 1, 0 + 1)
    if 'dt' in model_summary:
        t_tRF = t_tRF.astype(float) * model_summary['dt']
        axs[0].set_xlabel('Time')
    else:
        axs[0].set_xlabel('Frames')
    axs[0].fill_between(t_tRF, np.zeros_like(tRF), tRF)
    axs[0].vlines(t_tRF[peak_idxs], color='red', ymin=-np.max(np.abs(tRF)) * 1.1, ymax=np.max(np.abs(tRF)) * 1.1)
    axs[0].set_ylim(-np.max(np.abs(tRF)) * 1.1, np.max(np.abs(tRF)) * 1.1)

    axs[1].set(title='sRF')
    im = axs[1].imshow(sRF, cmap='bwr', vmin=-vmax, vmax=vmax, origin='lower')
    plt.colorbar(im, ax=axs[1])

    axs[2].set(title='sRF(max abs(w))')
    im = axs[2].imshow(w[np.argmax(np.sum(np.abs(w), axis=(1, 2))), :, :].T,
                       cmap='bwr', vmin=-vmax, vmax=vmax, origin='lower')
    plt.colorbar(im, ax=axs[2])

    axs[3].axis('off')

    if not model_summary:
        return

    axs[4].plot(model_summary['metric_train'], label='train', c='blue')
    if np.any(np.isfinite(model_summary['metric_dev'])):
        axs[4].plot(model_summary['metric_dev'], label='dev', c='red')
    axs[4].axhline(model_summary['metric_dev_opt'], label='dev-opt', ls='--', c='red')
    axs[4].axvline(model_summary['best_iteration'], ls='-', c='orange')
    axs[4].set(title=f"Metrics\n{model_summary['return_model']}",
               xlabel='Iteration', ylabel='Corrcoef')
    axs[4].legend()

    if model_summary is not None and len(model_summary) > 0:
        axs[5].plot(model_summary['cost_train'], label='train', c='blue')
        if np.any(np.isfinite(model_summary['cost_dev'])):
            axs[5].plot(model_summary['cost_dev'], label='dev', c='red', marker='.')
        axs[5].axvline(model_summary['best_iteration'], ls='-', c='orange')
        axs[5].set(title=f"Cost\n{model_summary['return_model']}",
                   xlabel='Iteration', ylabel='Cost')

        axs[6].loglog(np.arange(1, model_summary['cost_train'].size + 1),
                      model_summary['cost_train'], label='train', c='blue')
        axs[6].axvline(1 + model_summary['best_iteration'], ls='-', c='orange')
        axs[6].loglog(np.arange(1, model_summary['cost_dev'].size + 1),
                      model_summary['cost_dev'], label='dev', c='red', marker='.')

        axs[6].set(title=f"loglog-Cost\n{model_summary['return_model']}",
                   xlabel='Iteration', ylabel='Cost')

    if (quality_dict is not None) and ('score_permX' in quality_dict):
        plot_test_rf_quality(
            ax=axs[7], score_permX=quality_dict['score_permX'],
            score_test=quality_dict.get('score_test', np.nan),
            cc_test=quality_dict.get('corrcoef_test', None),
            cc_dev=quality_dict.get('corrcoef_dev', None),
        )
        title += f"\nquality={quality_dict['quality']}"
        title_kw = dict(backgroundcolor='green' if quality_dict['quality'] else 'red')
    else:
        title_kw = dict()
        axs[7].axis('off')

    for i, (y_type, ax) in enumerate(zip(['train', 'dev', 'test'], axbigs)):
        y_data = model_summary.get(f'y_{y_type}', None)
        y_pred = model_summary.get(f'y_pred_{y_type}', model_summary.get(f'y_pred', None))

        cc = quality_dict.get(f'corrcoef_{y_type}', np.nan)
        mse = quality_dict.get(f'mse_{y_type}', np.nan)

        if y_data is None or y_pred is None:
            ax.set_title(y_type + ' data not found')
            continue

        time = np.arange(0, y_data.size) * model_summary.get('dt', 1)

        ax.plot(time, y_data, label='data')
        ax.plot(time, y_pred, label='prediction', alpha=0.7)
        ax.legend(frameon=True)
        ax.set_title(f'{y_type} data and prediction: cc={cc:.2f}, mse={mse:.2f}')

    plt.tight_layout(rect=(0.01, 0.01, 0.99, 0.90), h_pad=0.2, w_pad=0.2)
    plt.suptitle(title, y=0.99, va='top', **title_kw)

    return fig, axs


def plot_test_rf_quality(score_permX, score_test, cc_test=None, cc_dev=None, ax=None, metric=''):
    """Plot test results"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    sns.boxplot(y=score_permX, ax=ax, width=0.8)
    sns.swarmplot(y=score_permX, ax=ax, color='k', alpha=0.5, label='perm X')
    ax.axhline(score_test, c='r', label=f'test {metric}', alpha=0.5, zorder=100)

    mu = np.mean(score_permX)
    std = np.std(score_permX)

    for i in range(0, 4):
        ax.axhline(mu + i * std, c='c', ls=':', alpha=0.5, zorder=0, label='_')

    title = ""
    if cc_dev is not None:
        title += f"cc_dev={cc_dev: .4f}"
    if cc_test is not None:
        title += f"\ncc_test={cc_test:.4f}"

    ax.set(title=title, ylabel='score')

    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(np.min([mu - 0.03, mu - 4 * std, score_test - 0.01]),
                np.max([mu + 0.03, mu + 4 * std, score_test + 0.01]))

    return ax
