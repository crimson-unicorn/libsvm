#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static char *line = NULL;
static int max_line_len;

void exit_with_help() {
	printf(
	"Usage: incocsvm-preprocess [options] sketch_train_file_base_name sketch_test_file_base_name output_train_file output_test_file\n"
	"options:\n"
	"-m number of models [MUST INPUT A CORRECT NUMBER > 0]: number of models\n"
	"-s the size of sketch [MUST INPUT A CORRECT NUMBER > 0]: this value must be the same as defined by SKETCH_SIZE in GraphChi\n"
	"-l [MUST PROVIDE]: total training instances, which should be consistent with the same option in incocsvm-train program\n"
	"-t [MUST PROVIDE]: total test instances\n"
	);
	exit(1);
}

/*!
 * @brief Read a single line from a file.
 */
static char* read_single_line(FILE *input) {
	int len;

	if(fgets(line, max_line_len, input) == NULL)
		return NULL;

	while(strrchr(line, '\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *)realloc(line, (size_t)max_line_len);
		len = (int)strlen(line);
		if(fgets(line+len, max_line_len-len, input) == NULL)
			break;
	}

	return line;
}

/*!
 * @brief This is the kernalized distance, which is Hamming distance, between two sketches a and b.
 */
int hamming_distance(unsigned long* a, unsigned long* b, int size) {
	int i;
	int d = 0;
	for (i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			++d;
		}
	}
	return d;
}

/*!
 * @brief Read the sketch file created by GraphChi.
 *
 */
unsigned long** read_sketches(const char *sketch_file_name, int num_model, int sketch_size) {
	int i, j, k;
	char *p, *endptr;
	FILE *fp = fopen(sketch_file_name, "r");
	if(fp == NULL) 
		return NULL;

	unsigned long** sketches = Malloc(unsigned long *, (unsigned long)num_model);
	max_line_len = 1024;
	line = Malloc(char, (unsigned long)max_line_len);

	fprintf(stderr, "\t\t======== Load Sketches in Memory ========\n");
	fprintf(stderr, "\t\tReading sketch file: %s\n", sketch_file_name);
	for (i = 0; i < num_model; i++) {
		read_single_line(fp);
		unsigned long* sketch = Malloc(unsigned long, (unsigned long)sketch_size);
		p = strtok(line, " ");
		sketch[0] = strtoul(p, &endptr, 10);

		for (j = 1; j < sketch_size; j++) {
			p = strtok(NULL, " ");
			sketch[j] = strtoul(p, &endptr, 10);
		}
		sketches[i] = sketch;
		fprintf(stderr, "\t\tSketch #%d: ", i);
		for (k = 0; k < sketch_size; k++) {
			fprintf(stderr, "%lu ", sketches[i][k]);
		}
		fprintf(stderr, "\n");
	}
	free(line);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	fprintf(stderr, "\t\t======== Done ========\n");

	return sketches;
}

/*!
 * @brief Once we are done with the sketch arrays, we should free them.
 */
void free_sketches(unsigned long** sketches, int num_model) {
	int i;
	for (i = 0; i < num_model; i++) {
		free(sketches[i]);
		sketches[i] = NULL;
	}
	free(sketches);
	sketches = NULL;
	return;
}


int main(int argc, char **argv) {
	FILE *output_train, *output_test;
	char sketches_train_base_name[1024];
	char sketches_test_base_name[1024];
	char sketches_name[1024];
	char instance_num[256];
	int i, j, k, d;
	int m = 0;
	int s = 0;
	int l = 0;
	int t = 0;
	fprintf(stderr, "======== User Input Information ========\n");
	// parse options
	for (i = 1; i < argc; i++) {
		if(argv[i][0] != '-') 
			break;
		++i;
		switch(argv[i-1][1]) {
			case 'm':
				m = atoi(argv[i]);
				fprintf(stderr, "Number of models: %d\n", m);
				if (m <= 0) {
					fprintf(stderr,"You need to provide the number of models.");
					exit_with_help();
				}
				break;
			case 's':
				s = atoi(argv[i]);
				fprintf(stderr, "Sketch Size: %d\n", s);
				if (s <= 0) {
					fprintf(stderr,"You need to provide the size of the sketch.");
					exit_with_help();
				}
				break;
			case 'l':
				l = atoi(argv[i]);
				fprintf(stderr, "Number of training instances: %d\n", l);
				if (l <= 0) {
					fprintf(stderr,"You must provide the total number of training instances.\n");
					exit_with_help();	
				}
				break;
			case 't':
				t = atoi(argv[i]);
				fprintf(stderr, "Number of testing instances: %d\n", t);
				if (t <= 0) {
					fprintf(stderr,"You must provide the total number of test instances.\n");
					exit_with_help();	
				}
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	unsigned long** train_instances[l];
	unsigned long** test_instances[t];

	if(i >= argc)
		exit_with_help();

	output_train = fopen(argv[i+2], "w");
	fprintf(stderr, "Train output file name: %s\n", argv[i+2]);
	if(output_train == NULL) {
		fprintf(stderr, "Canot open output training file %s\n", argv[i+2]);
		exit(1);
	}

	output_test = fopen(argv[i+3], "w");
	fprintf(stderr, "Test output file name: %s\n", argv[i+3]);
	if(output_test == NULL) {
		fprintf(stderr, "Canot open output test file %s\n", argv[i+3]);
		exit(1);
	}

	strcpy(sketches_train_base_name, argv[i]);
	fprintf(stderr, "Training sketch input base file name: %s\n", argv[i]);
	strcpy(sketches_name, sketches_train_base_name);

	for(j = 0; j < l; j++) {
		sprintf(instance_num, "%d", j);
		strcat(sketches_name, instance_num);
		strcat(sketches_name, ".txt");
		fprintf(stderr, "Training sketch input file name (%d): %s\n", j, sketches_name);
		train_instances[j] = read_sketches(sketches_name, m, s);
		if (train_instances[j] == NULL) {
			fprintf(stderr, "can't read instance %d from train file %s\n", j, sketches_name);
			exit(1);
		}
		strcpy(sketches_name, sketches_train_base_name);
	}

	strcpy(sketches_test_base_name, argv[i+1]);
	fprintf(stderr, "Testing sketch input base file name: %s\n", argv[i+1]);
	strcpy(sketches_name, sketches_test_base_name);

	for(j = 0; j < t; j++) {
		sprintf(instance_num, "%d", j);
		strcat(sketches_name, instance_num);
		strcat(sketches_name, ".txt");
		fprintf(stderr, "Testing sketch input file name (%d): %s\n", j, sketches_name);
		test_instances[j] = read_sketches(sketches_name, m, s);
		if (test_instances[j] == NULL) {
			fprintf(stderr, "can't read instance %d from test file %s\n", j, sketches_name);
			exit(1);
		}
		strcpy(sketches_name, sketches_test_base_name);
	}

	/* The format of the training output file:
	 * <label> 0:i 1:K(xi, x1) 2:K(xi, x2) ... L:K(xi, xL)
	 * <label> can be any value, ignored
	 * i: The ID of training instances, starting from 1.
	 *
	 * The arrangement of the file:
	 * training sketch 1, first model
	 * training sketch 2, first model
	 * ...
	 * training sketch 1, second model
	 * training sketch 2, second model
	 * ...
	 *
	 */
	fprintf(stderr, "======== End of User Input ========\n");
	fprintf(stderr, "======== Computing Training Input ========\n");
	for (i = 0; i < m; i++) {
		for (j = 0; j < l; j++) {
			fprintf(output_train, "%d 0:%d ", 1, j + 1);
			for (k = 0; k < l; k++) {
				d = hamming_distance(train_instances[j][i], train_instances[k][i], s);
				fprintf(stderr, "Training Hamming distance between %d and %d in model #%d: %d\n", j + 1, k + 1, i, d);
				fprintf(output_train, "%d:%d ", k + 1, d);
			}
			fprintf(output_train, "\n");
		}
	}

	if (ferror(output_train) != 0 || fclose(output_train) != 0)
		return -1;

	fprintf(stderr, "======== Done ========\n");
	fprintf(stderr, "======== Computing Test Input ========\n");

	/* The format of the test output file:
	 * <label> 0:? 1:K(x, x1) 2:K(x, x2) ... L:K(x, xL)
	 * <label> must be +1/-1
	 * ?: Can be any value, ignored
	 * K(x, xL): Hamming distance between the test sketch x and the L training sketch
	 * 
	 * The arrangement of the file:
	 * test sketch 1, first model
	 * test sketch 1, second model
	 * ...
	 * test sketch 2, first model
	 * ...
	 * 
	 */
	for (i = 0; i < t; i++) {
		for (j = 0; j < m; j++) {
			//TODO: TEST LABEL NEEDS TO BE DETERMINED SOMEHOW.
			fprintf(output_test, "%d 0:0 ", 1);
			for (k = 0; k < l; k++) {
				d = hamming_distance(test_instances[i][j], train_instances[k][j], s);
				fprintf(stderr, "Test Hamming distance between testing %d and training %d in model #%d: %d\n", i + 1, k + 1, j, d);
				fprintf(output_test, "%d:%d ", k + 1, d);
			}
			fprintf(output_test, "\n");
		}
	}
	fprintf(stderr, "======== Done ========\n");

	for(j = 0; j < l; j++) {
		free_sketches(train_instances[j], m);
	}

	for(j = 0; j < t; j++) {
		free_sketches(test_instances[j], m);
	}

	if (ferror(output_test) != 0 || fclose(output_test) != 0)
		return -1;

	return 0;
}