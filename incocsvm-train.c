#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"This training program is modified for a specific use of libSVM.\n"
	"We attempt to make as few changes as possible, although understandably the program is made less flexible but tailored for our purposes.\n"
	"Hence, to minimize unexpected behavior, make sure you follow the guideline below.\n"
	"==========================================================================\n"
	"Usage: incocsvm-train [options] training_set_file [model_file]\n"
	"options [RESTRICTED]:\n"
	"-s svm_type [DO NOT CHANGE THE DEFAULT OPTION]: set type of SVM (default 2)\n"
	"	0 -- C-SVC		(multi-class classification)\n"
	"	1 -- nu-SVC		(multi-class classification)\n"
	"	2 -- one-class SVM [OUR CHOICE OF SVM FOR OUTLIER DETECTION]\n"
	"	3 -- epsilon-SVR	(regression)\n"
	"	4 -- nu-SVR		(regression)\n"
	"-t kernel_type [DO NOT CHANGE THE DEFAULT OPTION]: set type of kernel function (default 4)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file) [DATA IS KERNALIZED TO THE HAMMING SPACE]\n"
	"-d degree [NOT USED]: set degree in kernel function (default 3)\n"
	"-g gamma [NOT USED]: set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 [NOT USED]: set coef0 in kernel function (default 0)\n"
	"-c cost [NOT USED]: set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon [NOT USED]: set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight [NOT USED]: set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	"-l [MUST PROVIDE]: total training instances\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
/* We do not use a global svm_problem struct or svm_model struct pointer or struct svm_node pointer in our case. */
// struct svm_problem prob;		// set by read_problem
// struct svm_model *model;
// struct svm_node *x_space;
int cross_validation;
int nr_fold;
int num_model;

/* Set of SVM problems for each model. This is set by read_problem. */
struct svm_problem **prob_set;
/* An array of svm_model pointers, each of which points to a model at a certain point. */
struct svm_model **profile;
struct svm_node **x_spaces;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input) {
	int len;

	if(fgets(line, max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL) {
		max_line_len *= 2;
		line = (char *) realloc(line, (size_t)max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len, max_line_len-len, input) == NULL)
			break;
	}
	return line;
}

int main(int argc, char **argv) {
	char input_file_name[1024];
	char model_file_name[1024];
	char model_file_name_copy[1024];
	const char *error_msg;
	int i;
	char model_num[256];

	parse_command_line(argc, argv, input_file_name, model_file_name);
	strcat(model_file_name_copy, model_file_name);
	read_problem(input_file_name);

	profile = Malloc(struct svm_model *, (unsigned long)num_model);

	for(i = 0; i < num_model; i++) {
		error_msg = svm_check_parameter(prob_set[i], &param);

		if(error_msg) {
			fprintf(stderr,"ERROR AT MODEL %d: %s\n", i, error_msg);
			exit(1);
		}

		if(cross_validation) {
			do_cross_validation();
		}
		else {
			profile[i] = svm_train(prob_set[i], &param);

			sprintf(model_num, "%d", i);
			strcat(model_file_name, model_num);
			if (svm_save_model(model_file_name, profile[i])) {
				fprintf(stderr, "can't save model %d to file %s\n", i, model_file_name);
				exit(1);
			}
			strcpy(model_file_name, model_file_name_copy);
			svm_free_and_destroy_model(&profile[i]);
		}
		free(prob_set[i]->y);
		free(prob_set[i]->x);
		free(prob_set[i]);
		free(x_spaces[i]);
	}
	free(prob_set);
	free(profile);
	free(x_spaces);
	free(line);
	svm_destroy_param(&param);

	return 0;
}

void do_cross_validation() {
	int i, n;
	int total_correct = 0;
	// double total_error = 0;
	// double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, (unsigned long)param.num_train);

	for (n = 0; n < num_model; n++) {
		svm_cross_validation(prob_set[n], &param, nr_fold, target);
		/* We do not need to use the commented code below since we are dealing with OCSVM only. */
		// if(param.svm_type == EPSILON_SVR ||
		//    param.svm_type == NU_SVR)
		// {
		// 	for(i=0;i<prob.l;i++)
		// 	{
		// 		double y = prob.y[i];
		// 		double v = target[i];
		// 		total_error += (v-y)*(v-y);
		// 		sumv += v;
		// 		sumy += y;
		// 		sumvv += v*v;
		// 		sumyy += y*y;
		// 		sumvy += v*y;
		// 	}
		// 	printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		// 	printf("Cross Validation Squared correlation coefficient = %g\n",
		// 		((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
		// 		((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
		// 		);
		// }
		// else
		// {
		for(i = 0; i < param.num_train; i++) {
			if(target[i] == prob_set[n]->y[i])
				++total_correct;
			printf("Cross Validation Accuracy = %g%%\n", 100.0 * total_correct / param.num_train);
		}
	}
	free(target);
}

/*!
 * @change We include an extra argument "num_train".
 */
void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name) {
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = ONE_CLASS;
	param.kernel_type = PRECOMPUTED;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.num_train = 0;
	cross_validation = 0;

	// parse options
	for(i = 1; i < argc; i++) {
		if(argv[i][0] != '-') break;
		if(++i >= argc)
			exit_with_help();
		switch(argv[i-1][1]) {
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2) {
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label, sizeof(int)*(unsigned long)param.nr_weight);
				param.weight = (double *)realloc(param.weight, sizeof(double)*(unsigned long)param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			case 'l':
				param.num_train = atoi(argv[i]);
				if (param.num_train <= 0) {
					fprintf(stderr,"You must provide the total number of training instances.\n");
					exit_with_help();	
				}
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if(i >= argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i < argc - 1)
		strcpy(model_file_name,argv[i+1]);
	else {
		char *p = strrchr(argv[i], '/');
		if(p == NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name, "%s.model", p);
	}
}

// read in a set of problems (in svmlight format)
void read_problem(const char *filename) {
	int max_index, inst_max_index, i, total_line, n;
	size_t elements, j;
	FILE *fp = fopen(filename, "r");
	char *endptr;
	char *idx, *val, *label;
	struct svm_problem *prob;
	struct svm_node *x_space;

	if(fp == NULL) {
		fprintf(stderr,"can't open input file %s\n", filename);
		exit(1);
	}

	/* This is tailored specifically for our cases since we know how many training instances for each model. 
	 * The training file itself contains training instances for all models. 
	 */
	elements = (unsigned long)param.num_train * (unsigned long)param.num_train + 2 * (unsigned long)param.num_train; /* We also know the total number of elements. */
	total_line = 0;

	max_line_len = 1024;
	line = Malloc(char, (unsigned long)max_line_len);

	while(readline(fp) != NULL) {
		++total_line;
	}
	rewind(fp);

	/* num_model is the number of models that we are building. 
	 * This corresponds to the number of svm problems that we will see. */
	num_model = total_line / param.num_train;
	prob_set = Malloc(struct svm_problem *, (unsigned long)num_model);
	x_spaces = Malloc(struct svm_node *, (unsigned long)num_model);

	for (n = 0; n < num_model; n++) {
		
		prob = Malloc(struct svm_problem, 1);
		prob_set[n] = prob;

		prob->l = param.num_train;
		prob->y = Malloc(double, (unsigned long)prob->l);
		prob->x = Malloc(struct svm_node *, (unsigned long)prob->l);

		x_space = Malloc(struct svm_node, elements);
		x_spaces[n] = x_space;

		max_index = 0;
		j = 0;
		for(i = 0; i < prob->l; i++) {
			inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
			readline(fp);
			prob->x[i] = &x_space[j];
			label = strtok(line," \t\n");
			if(label == NULL) // empty line
				exit_input_error(i+1);

			prob->y[i] = strtod(label, &endptr);
			if(endptr == label || *endptr != '\0')
				exit_input_error(i+1);

			while(1) {
				idx = strtok(NULL, ":");
				val = strtok(NULL, " \t");

				if(val == NULL)
					break;

				errno = 0;
				x_space[j].index = (int) strtol(idx, &endptr, 10);
				if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
					exit_input_error(i+1);
				else
					inst_max_index = x_space[j].index;

				errno = 0;
				x_space[j].value = strtod(val, &endptr);
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);

				++j;
			}

			if(inst_max_index > max_index)
				max_index = inst_max_index;
			x_space[j++].index = -1;
		}

		if(param.gamma == 0 && max_index > 0)
			param.gamma = 1.0 / max_index;

		if(param.kernel_type == PRECOMPUTED) {
			for(i = 0; i < prob->l; i++) {
				if (prob->x[i][0].index != 0) {
					fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
					exit(1);
				}
				if ((int)prob->x[i][0].value <= 0 || (int)prob->x[i][0].value > max_index) {
					fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
					exit(1);
				}
			}
		}
	}

	fclose(fp);
}
