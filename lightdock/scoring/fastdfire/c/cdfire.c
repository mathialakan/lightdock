#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PyInt_AsUnsignedLongMask PyLong_AsUnsignedLongMask
#include <Python.h>
#include "structmember.h"
#include "numpy/arrayobject.h"


/**
 *
 * DFIRE distances
 *
 **/
static unsigned int dist_to_bins[50] = {
         1,  1,  1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 14, 15, 15, 16,
        16, 17, 17, 18, 18, 19, 19, 20, 20, 21,
        21, 22, 22, 23, 23, 24, 24, 25, 25, 26,
        26, 27, 27, 28, 28, 29, 29, 30, 30, 31};


/**
 *
 * Auxiliary function
 *
 **/
int compare(const void *a, const void *b) {
    const unsigned int *da = (const unsigned int *) a;
    const unsigned int *db = (const unsigned int *) b;

    return (*da > *db) - (*da < *db);
}

/*
 *  Acceleratble code using GPU or OpenACC etc.
 *
 */

void compute_acc(double ** rec_array, double ** lig_array, unsigned int rec_len, unsigned int lig_len, unsigned int *indexes_len,
                    unsigned long long * rec_obj, unsigned long long * lig_obj, unsigned int ** interface_receptor,
                    unsigned int ** interface_ligand, unsigned int ** array, double interface_cutoff, unsigned int *interface_len,
		    double * dfire_en_array, double *energy){
   
   unsigned int i, j, m, n = 0, d; //, indexes_len = 0;
   unsigned int atoma, atomb, dfire_bin ;
   unsigned int tot_len  = rec_len*lig_len;
   //size_t bytes1 = tot_len*sizeof(unsigned int);
   double * dist = malloc(tot_len*sizeof(double));
   unsigned int * indexes = malloc(3*tot_len*sizeof(unsigned int));
   
   //---------------------------------------------------------
   //Computing distance 
   double sub1, sub2, sub3;
   for (i = 0; i < rec_len; i++) {
        for (j = 0; j < lig_len; j++) {
            sub1 = rec_array[i][0] - lig_array[j][0];
            sub2 = rec_array[i][1] - lig_array[j][1];
            sub3 = rec_array[i][2] - lig_array[j][2];
            dist[n++] = sub1*sub1 + sub2*sub2 + sub3*sub3;
        }
    }

   //Selecting the nearest atoms
   n = 0;
   for(i = 0; i < tot_len; i++){
       if (dist[i] <= 225.) {
                indexes[n++] = i/rec_len;
                indexes[n++] = i%rec_len;
                indexes[n++] = (sqrt(dist[i])*2.0 - 1.0);
                (*indexes_len)++;
            }
   }

   indexes = realloc(indexes, n*sizeof(unsigned int));
    // free(*interface_receptor);

   //---------------------------------------------------------
   // Computing receptor_ligand interface
   size_t bytes = (*indexes_len)*sizeof(unsigned int);
   *array = malloc(bytes);
   *interface_receptor = malloc(bytes);
   *interface_ligand = malloc(bytes);

   for(n = m = 0; n < (*indexes_len); n++){ 
       i = indexes[m++];
       j = indexes[m++];
       d = indexes[m++];

       if (d <= interface_cutoff) {
           (*interface_receptor)[(*interface_len)] = i;
           (*interface_ligand)[(*interface_len)++] = j;
       }
       atoma = rec_obj[i];
       atomb = lig_obj[j];
       dfire_bin = dist_to_bins[d] - 1;
    
       (*array)[n] = atoma*168*20 + atomb*20 + dfire_bin;
   }
     
    //---------------------------------------------------------
    // Computing energy 
    // unsigned int index;
    for (n = 0; n < (*indexes_len); n++) {
      // index = (*array)[n];
        (*energy) += dfire_en_array[n];
    }

//        free(*array);
   
        free(indexes);
  //      printf("indexes_len: %d\n", (*indexes_len));
}



/**
 *
 * calculate_dfire C implementation
 *
 **/
static PyObject * cdfire_calculate_dfire(PyObject *self, PyObject *args) {
    PyObject *receptor, *ligand, *dfire_energy, *receptor_coordinates, *ligand_coordinates;
    PyObject *take, *array_object,  *result = NULL;
    PyArrayObject *df_en_array;
    //PyObject *df_en_array;
    unsigned int  indexes_len, interface_len, *array=NULL, *interface_receptor=NULL, *interface_ligand=NULL;// *indexes;
    double interface_cutoff, energy, *dfire_en_array;
    npy_intp dims[1];

    interface_cutoff = 3.9;
    energy = 0.;
    interface_len = 0;

    if (PyArg_ParseTuple(args, "OOOOO|d", &receptor, &ligand, &dfire_energy, &receptor_coordinates, &ligand_coordinates, &interface_cutoff)) {

        PyObject *tmp0, *tmp1, *tmp2, *tmp3;
	unsigned int rec_len, lig_len;
    	double  **rec_array, **lig_array;
    	npy_intp dims[2];
    	npy_intp dims_1[1];

    	indexes_len = 0;

    	tmp0 = PyObject_GetAttrString(receptor_coordinates, "coordinates");
    	tmp1 = PyObject_GetAttrString(ligand_coordinates, "coordinates");

    	rec_len = PySequence_Size(tmp0);
    	lig_len = PySequence_Size(tmp1);

    	dims[1] = 3;
    	dims[0] = rec_len;
    	PyArray_AsCArray((PyObject **)&tmp0, (void **)&rec_array, dims, 2, PyArray_DescrFromType(NPY_DOUBLE));

    	dims[0] = lig_len;
    	PyArray_AsCArray((PyObject **)&tmp1, (void **)&lig_array, dims, 2, PyArray_DescrFromType(NPY_DOUBLE));

       // * Computation of Euclidean distances and selection of nearest atoms
      // indexes = malloc(3*rec_len*lig_len*sizeof(unsigned int));
	//euclidean_dist(rec_array, lig_array, rec_len, lig_len, &indexes, &indexes_len);

//	PyArray_Free(tmp0, rec_array);
//    	PyArray_Free(tmp1, lig_array);
    	Py_DECREF(tmp0);
    	Py_DECREF(tmp1);


        // Do not need to free rec_objects and lig_objects
        tmp2 = PyObject_GetAttrString(receptor, "objects");
        int rec_obj_len = PySequence_Size(tmp2);
        tmp2 = PySequence_Fast(tmp2, "");
        Py_DECREF(tmp2);
	//rec_objects = PySequence_Fast_ITEMS(tmp1);
        //Py_DECREF(tmp1);

	dims_1[0] = rec_obj_len;
        unsigned long long * rec_obj;
        PyArray_AsCArray( (PyObject **)&tmp2, (void *)&rec_obj, dims_1, 1, PyArray_DescrFromType(NPY_INT));

        tmp3 = PyObject_GetAttrString(ligand, "objects");
        int lig_obj_len = PySequence_Size(tmp3);
        tmp3 = PySequence_Fast(tmp3, "");
        Py_DECREF(tmp3);
        //lig_objects = PySequence_Fast_ITEMS(tmp3);
        //Py_DECREF(tmp3);
        dims_1[0] = lig_obj_len;
        unsigned long long * lig_obj;
        PyArray_AsCArray((PyObject **) &tmp3, (void *)&lig_obj, dims_1, 1, PyArray_DescrFromType(NPY_INT));

        //df_en_array = PyArray_FROM_OTF(dfire_energy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	//dfire_en_array = (double*)PyArray_DATA(df_en_array);
 
        dims[0] = indexes_len;
        tmp0 = PyImport_ImportModule("numpy");
        take = PyObject_GetAttrString(tmp0, "take");
        Py_DECREF(tmp0);
        array_object = PyArray_SimpleNewFromData(1, dims, NPY_UINT, array);
        df_en_array = (PyArrayObject *)PyObject_CallFunctionObjArgs(take, dfire_energy, array_object, NULL);
        dfire_en_array = PyArray_GETPTR1(df_en_array, 0);

        //printf("size: %ld \n", sizeof(dfire_en_array));

	compute_acc(rec_array, lig_array, rec_len, lig_len, &indexes_len,
                       rec_obj, lig_obj,  &interface_receptor, &interface_ligand, &array, interface_cutoff, &interface_len, dfire_en_array, &energy);

        PyArray_Free(tmp0, rec_array);
        PyArray_Free(tmp1, lig_array);
        PyArray_Free(tmp2, rec_obj);
        PyArray_Free(tmp3, lig_obj);

        
        free(array);
        //free(indexes);

        Py_DECREF(df_en_array);
        Py_DECREF(array_object);
        Py_DECREF(take);
    }

    dims[0] = interface_len;

    interface_receptor = realloc(interface_receptor, interface_len*sizeof(unsigned int));
    interface_ligand = realloc(interface_ligand, interface_len*sizeof(unsigned int));

    result = PyTuple_New(3);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble((energy*0.0157 - 4.7)*-1));
    PyTuple_SET_ITEM(result, 1, PyArray_SimpleNewFromData(1, dims, NPY_UINT, interface_receptor));
    PyTuple_SET_ITEM(result, 2, PyArray_SimpleNewFromData(1, dims, NPY_UINT, interface_ligand));

    return result;
}


/**
 *
 * Module methods table
 *
 **/
static PyMethodDef module_methods[] = {
    {"calculate_dfire", (PyCFunction)cdfire_calculate_dfire, METH_VARARGS, "calculate_dfire C implementation"},
    {NULL}
};


/**
 *
 * Initialization function
 *
 **/
static struct PyModuleDef cdfire =
{
    PyModuleDef_HEAD_INIT,
    "cdfire",
    "",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_cdfire(void) {
    import_array();
    return PyModule_Create(&cdfire);
}
