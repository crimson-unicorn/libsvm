#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct svm_node *x;
int max_nr_attr = 64;
int total_model = 0;

struct svm_model *model;
struct svm_model **profile;

int predict_probability=0;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input) {
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line, '\n') == NULL) {
		max_line_len *= 2;
		line = (char *)realloc(line, (size_t)max_line_len);
		len = (int)strlen(line);
		if(fgets(line+len, max_line_len-len, input) == NULL)
			break;
	}
	return line;
}

void exit_input_error(int line_num) {
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	int counter = 0;
	int model_to_use;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	/* All models should have the same SVM type and nr_class. */
	int svm_type = svm_get_svm_type(profile[0]); /* We will use the value from the first model. */
	int nr_class=svm_get_nr_class(profile[0]);
	double *prob_estimates=NULL;
	int k, j;

	/* We check that this is the case: */
	assert(total_model > 0);
	for (k = 0; k < total_model; k++) {
		assert(svm_get_svm_type(profile[k]) == svm_type);
		assert(svm_get_nr_class(profile[k]) == nr_class);
	}

	/* The following if statement should always be false in our case. */
	if (predict_probability) {
		assert(0); /* We make sure this block of code never runs. */
		if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else {
			int *labels = (int *)malloc((unsigned long)nr_class * sizeof(int));
			svm_get_labels(model, labels);
			prob_estimates = (double *)malloc((unsigned long)nr_class * sizeof(double));
			fprintf(output, "labels");
			for(j = 0; j < nr_class; j++)
				fprintf(output, " %d", labels[j]);
			fprintf(output, "\n");
			free(labels);
		}
	}

	max_line_len = 1024;
	line = (char *)malloc((unsigned long)max_line_len * sizeof(char));
	while(readline(input) != NULL) {
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line, " \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label, &endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1) {
			if(i >= max_nr_attr - 1) {	// need one more for index = -1
				max_nr_attr *= 2;
				x = (struct svm_node *)realloc(x, (unsigned long)max_nr_attr * sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx, &endptr, 10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val, &endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;

		/* We pick which model to run here. */
		model_to_use = counter % total_model; /* Currently, we use round robin. */
		model = profile[model_to_use];

		// if (predict_probability && (svm_type == C_SVC || svm_type == NU_SVC)) {
			// predict_label = svm_predict_probability(model,x,prob_estimates);
			// fprintf(output,"%g",predict_label);
			// for(j=0;j<nr_class;j++)
			// 	fprintf(output," %g",prob_estimates[j]);
			// fprintf(output,"\n");
		// } else {
		predict_label = svm_predict(model, x);
		fprintf(output, "%.17g\n", predict_label);
		// }

		if(predict_label == target_label)
			++correct;
		error += (predict_label - target_label) * (predict_label - target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label * predict_label;
		sumtt += target_label * target_label;
		sumpt += predict_label * target_label;
		++total;
		++counter;
	}
	if (svm_type == NU_SVR || svm_type == EPSILON_SVR) {
		assert(0); /* This assert should never run. */
		info("Mean squared error = %g (regression)\n", error / total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	} else
		info("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct / total * 100, correct, total);
	
	if(predict_probability)
		free(prob_estimates);
}

void exit_with_help() {
	printf(
	"Usage: incocsvm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-n number of models [MUST INPUT A CORRECT NUMBER > 0]: number of models from training\n"
	"-b probability_estimates [MUST BE 0]: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

int main(int argc, char **argv) {
	FILE *input, *output;
	char model_base_name[1024];
	char model_name[1024];
	char model_num[256];
	int i, j;
	int n = 0;
	// parse options
	for (i = 1; i < argc; i++) {
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1]) {
			case 'n':
				n = atoi(argv[i]);
				if (n <= 0) {
					fprintf(stderr,"You need to provide the number of models.");
					exit_with_help();
				}
				total_model = n;
				break;
			case 'b':
				predict_probability = atoi(argv[i]);
				if (predict_probability != 0) {
					fprintf(stderr,"Predict probability must be 0.");
					exit_with_help();
				}
				break;
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	if(i >= argc-2)
		exit_with_help();

	input = fopen(argv[i], "r");
	if(input == NULL) {
		fprintf(stderr,"can't open input file %s\n", argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2], "w");
	if(output == NULL) {
		fprintf(stderr,"can't open output file %s\n", argv[i+2]);
		exit(1);
	}

	strcpy(model_base_name, argv[i+1]);
	profile = Malloc(struct svm_model *, (unsigned long)n);

	for (j = 0; j < n; j++) {
		strcpy(model_name, model_base_name);
		sprintf(model_num, "%d", j);
		strcat(model_name, model_num);
		if((model = svm_load_model(model_name)) == 0) {
			fprintf(stderr,"can't open model file %s\n", model_name);
			exit(1);
		}

		profile[j] = model;

		if(predict_probability) {
			if(svm_check_probability_model(model) == 0) {
				fprintf(stderr,"Model does not support probabiliy estimates\n");
				exit(1);
			}
		} else {
			if(svm_check_probability_model(model) != 0)
				info("Model supports probability estimates, but disabled in prediction.\n");
		}
	}

	x = (struct svm_node *)malloc((unsigned long)max_nr_attr * sizeof(struct svm_node));
	
	predict(input, output);

	for (j = 0; j < n; j++) {
		svm_free_and_destroy_model(&profile[j]);
	}

	free(profile);
	free(x);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}
