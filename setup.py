from setuptools import setup, Extension

TREEPATH = "granite/tree_shap"

setup(
    ext_modules=[Extension('granite.treeshap', sources=[f"{TREEPATH}/main.cpp"], 
                           extra_compile_args=["-std=c++11"],
                           depends=[ f"{TREEPATH}/leaf_treeshap.hpp", f"{TREEPATH}/progressbar.hpp",
                                     f"{TREEPATH}/recursive_treeshap.hpp", f"{TREEPATH}/utils.hpp",
                                     f"{TREEPATH}/waterfall_treeshap.hpp"])]

)