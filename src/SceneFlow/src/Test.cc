#include "SceneFlow.h"
#include <glog/logging.h>

using namespace std;

int main() {
    std::string py_path; 	/*!< Path to be included to the environment variable PYTHONPATH */
	std::string module_name; /*!< Detailed description after the member */
	std::string class_name; /*!< Detailed description after the member */
    std::string get_optical_flow; 	/*!< Detailed description after the member */

    std::string strSettingsFile = "/mnt/SceneFlow/OpticalFlow.yaml";
    cv::FileStorage fs(strSettingsFile.c_str(), cv::FileStorage::READ);
    fs["py_path"] >> py_path;
    fs["module_name"] >> module_name;
    fs["class_name"] >> class_name;
    fs["get_optical_flow"] >> get_optical_flow;

    LOG(INFO) << "------py_path: "<< py_path;
    LOG(INFO) << "------module_name: "<< module_name;
    LOG(INFO) << "------class_name: "<< class_name;
    LOG(INFO) << "------get_optical_flow: "<< get_optical_flow;

    std::string x;
    setenv("PYTHONPATH", py_path.c_str(), 1);
    x = getenv("PYTHONPATH");

    // ----初始化
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        printf("初始化失败");
        PyErr_Print();
        std::exit(1);
    }
    PyRun_SimpleString("import sys");
    string dir = "sys.path.append('/mnt/SceneFlow/src/RAFT/core')";
    PyRun_SimpleString(dir.c_str());

    // const static int numpy_initialized =  init_numpy();
    // LOG(INFO) << numpy_initialized << endl;

    // -----导入py文件
    PyObject* pModule = PyImport_ImportModule(module_name.c_str());  //导入模型

    if (pModule == nullptr)
    {                                                                                
        PyErr_Print();
        std::exit(1);
    }
    assert(pModule != NULL);

    // ----模块的字典列表
    LOG(INFO) << "------pDict";
    PyObject* pDict = PyModule_GetDict(pModule);
    if (pDict == nullptr)
    {
        PyErr_Print();
        std::exit(1);
    }
    assert(pDict != NULL);

    // ----导入已导入模块中的方法或类
    // LOG(INFO) << "------pClass";
    // PyObject* pClass = PyDict_GetItemString(pDict, "Optical");
    // if (pClass == nullptr)
    // {
    //     PyErr_Print();
    //     std::exit(1);
    // }
    // assert(pClass != NULL);

    // ----创建实例
    // LOG(INFO) << "------pIns";
    // PyObject* pInstance = PyInstanceMethod_New(pClass);
    // if (pInstance == nullptr)
    // {
    //     PyErr_Print();
    //     std::exit(1);
    // }
    // assert(pInstance != NULL);

    LOG(INFO) << "------pFunc";
    PyObject *pFunc = PyObject_GetAttrString(pModule, "optical");
    if (pFunc == nullptr)
    {
        PyErr_Print();
        std::exit(1);
    }
    assert(pFunc != NULL);

    LOG(INFO) << "------py_image";
    // PyObject* args = PyTuple_New(2);
    // PyObject* py_image1 = cvt->toNDArray(in_image1.clone());
    // PyObject* py_image2 = cvt->toNDArray(in_image2.clone());
    // PyTuple_SetItem(args, 0 , py_image1);
    // PyTuple_SetItem(args, 1 , py_image2);
    // assert(py_image1 != NULL);
    // assert(py_image2 != NULL);
    // LOG(INFO) << "0";
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, Py_BuildValue("i", 3));

    
    LOG(INFO) << "------py_optical";
    // PyObject* py_optical = PyObject_CallMethod(pInstance, 
    //                                             const_cast<char*>(get_optical_flow.c_str()), 
    //                                         "(OOO)", pInstance, py_image1, py_image1);

    // PyObject* args = PyTuple_New(2);
    // PyTuple_SetItem(args, 0, py_image1);
    // PyTuple_SetItem(args, 1, py_image2);
    // LOG(INFO) << "1";
    PyObject* py_optical = PyObject_CallObject(pFunc, args);

    LOG(INFO) << "2";
    // PyCallable_Check(py_optical);
    if (py_optical == nullptr)
    {
        PyErr_Print();
        std::exit(1);
    }
    assert(py_optical != nullptr);
}