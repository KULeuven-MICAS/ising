import pathlib
import matplotlib.pyplot as plt
import numpy as np

from ising.postprocessing.helper_functions import return_data

def plot_energies_on_figure(energies: np.ndarray, label: str | None = None):
    plt.plot(energies, label=label)


def plot_energies(fileName: pathlib.Path, save: bool = True, save_folder: pathlib.Path = "."):
    energies, best_energy, solver_name = (
        return_data(fileName=fileName, data="energies"),
        return_data(fileName, data="solution_energy"),
        return_data(fileName, data="solver"),
    )

    plt.figure()
    plot_energies_on_figure(energies)
    plt.title(f"Best energy: {best_energy}")
    plt.xlabel("iteration")
    plt.ylabel("Energy")

    if save:
        plt.savefig(f"{save_folder}/{solver_name}_energy.png")

    plt.show()


def plot_energies_multiple(fileName_list: list[pathlib.Path], save: bool = True, save_folder: pathlib.Path = "."):
    plt.figure()
    title = ""
    for fileName in fileName_list:
        energies, best_energy, solver_name = (
            return_data(fileName=fileName, data="energies"),
            return_data(fileName, data="solution_energy"),
            return_data(fileName, data="solver"),
        )

        plot_energies_on_figure(energies, label=f"{solver_name} (Best: {best_energy})")
        title += solver_name + ", "
        plt.legend()
    plt.title(f"Energies of {title[:-2]}")
    plt.xlabel("iteration")
    plt.ylabel("Energy")
    if save:
        plt.savefig(save_folder + "/multiple_energies.png")
    plt.show()


def plot_energy_dist(fileName_list: list[pathlib.Path], save: bool = True, save_folder: pathlib.Path = "."):
    data = dict()
    solver = ""
    for fileName in fileName_list:
        best_energy = return_data(fileName=fileName, data="solution_energy")
        num_iter = return_data(fileName=fileName, data="num_iterations")
        solvername = return_data(fileName=fileName, data="solver")
        if solver == "":
            solver = solvername
        if solver == solvername:
            if num_iter not in data:
                data[num_iter] = [best_energy]
            else:
                data[num_iter].append(best_energy)
        else:
            print("Only one solver is allowed")
    avg_best_energies = []
    std_best_energies = []
    num_iters = []
    for num_iter, best_energies in data.items():
        all_best_energies = np.array(best_energies)
        avg_best_energies.append(np.mean(all_best_energies, axis=0))
        std_best_energies.append(np.std(all_best_energies, axis=0))
        num_iters.append(num_iter)

    plt.figure()
    plt.errorbar(num_iters, avg_best_energies, yerr=std_best_energies, fmt="-o")
    plt.title(f"Average Best Energy of {solver} with Standard Deviation")
    plt.xlabel("iteration")
    plt.ylabel("Best Energy")
    plt.legend()
    if save:
        plt.savefig(f"{save_folder}/{solver}_best_energy_distribution.png")
    plt.show()


def plot_energy_dist_multiple_solvers(
    fileName_list: list[pathlib.Path], save: bool = True, save_folder: pathlib.Path = "."
):
    data = dict()
    for fileName in fileName_list:
        best_energy = return_data(fileName=fileName, data="solution_energy")
        num_iter = return_data(fileName=fileName, data="num_iterations")
        solvername = return_data(fileName=fileName, data="solver")
        if solvername not in data:
            data[solvername] = {}
        if num_iter not in data[solvername]:
            data[solvername][num_iter] = [best_energy]
        else:
            data[solvername][num_iter].append(best_energy)

    plt.figure()
    for solver, iter_data in data.items():
        avg_best_energies = []
        std_best_energies = []
        num_iters = []
        for num_iter, best_energies in iter_data.items():
            all_best_energies = np.array(best_energies)
            avg_best_energies.append(np.mean(all_best_energies, axis=0))
            std_best_energies.append(np.std(all_best_energies, axis=0))
            num_iters.append(num_iter)
        plt.errorbar(
            num_iters, avg_best_energies, yerr=std_best_energies, fmt="-o", label=f"{solver} Average Best Energy"
        )

    plt.title("Average Best Energy with Standard Deviation for Multiple Solvers")
    plt.xlabel("iteration")
    plt.ylabel("Best Energy")
    plt.legend()
    if save:
        plt.savefig(f"{save_folder}/best_energy_distribution_multiple_solvers.png")
    plt.show()
