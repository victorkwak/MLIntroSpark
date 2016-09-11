import java.io.*;
import java.util.ArrayList;
import java.util.TreeMap;


public class GenerateScore {
    public static void main(String[] args) {
        try {
            String referenceFile = null;
            String redditData = null;
            if (args.length < 1 || args[0].equals("test")) {
                referenceFile = "data/test/Accuracy.txt";
                redditData = "data/test/RedditData";

            } else if (args[0].equals("eval")) {
                referenceFile = "data/eval/eval_Accuracy.txt";
                redditData = "data/eval/evalReddit";
            } else {
                System.out.println("Unknown option: " + args[0]);
                System.exit(1);
            }

            final double NUM_TRAINING_DATA = .90;

            File redditDataFolder = new File(redditData);

            File[] filenames = redditDataFolder.listFiles(new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.matches(".+\\.TITLE");
                }
            });


            System.out.println("CHECKING TRAINING DATA FOLDER");
            for (int i = 0; i < filenames.length; i++) {
                System.out.println(filenames[i]);
            }

            //load all titles into memory, in a mapping with whatever subreddit it came from
            TreeMap<String, ArrayList<String>> allTitles = new TreeMap<>();
            if (filenames == null) {
                System.out.println("ERROR");
                System.exit(1);
            }
            for (File filename : filenames) {
                if (!filename.toString().matches(".*\\w+\\.TITLE")) {
                    continue;
                }
                ArrayList<String> documents = new ArrayList<>();
                String category = filename.toString().substring(21, filename.toString().length() - 6);


                BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
                String currentLine;
                while ((currentLine = bufferedReader.readLine()) != null && !currentLine.equals("")) {
                    documents.add(currentLine);
                }
                allTitles.put(category, documents);
            }

            System.out.println("ADDING TRAINING DATA TO CLASSIFIER");
            //use NUM_TRAINING_DATA percentage of each class' titles as training data for the classifier
            NaiveBayesClassifier rc = new NaiveBayesClassifier();

            for (String s : allTitles.keySet()) {
                int totalDocs = allTitles.get(s).size();
                for (int i = 0; i < NUM_TRAINING_DATA * totalDocs; i++) {
                    try {
                        rc.addTrainingDocument(allTitles.get(s).remove(0), s);
                    } catch (Exception e) {
                        throw new TrainingDocException();
                    }
                }
            }
            System.out.println("The number of documents in your classifier is: " + rc.getTotalDocumentCount());
            if (rc.getTotalDocumentCount() != 4500)
                throw new TrainingDocException("Wrong Doc Count");

            System.out.println("CLASSIFYING");
            int[][] confusionMatrix = new int[filenames.length][filenames.length];
            for (int i = 0; i < filenames.length; i++)
                for (int j = 0; j < filenames.length; j++)
                    confusionMatrix[i][j] = 0;
        /*
        class denumeration:
        0 ama
        1 askengineers
        2 economics
        3 fitness
        4 showerthoughts
         */
            //BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("eval/test.txt"));
            int success = 0;
            int fail = 0;
            for (String s : allTitles.keySet()) {
                int denumeratedSubreddit = 0;
                for (int i = 0; i < filenames.length; i++) {
                    if (s.equals(filenames[i].toString().substring(21, filenames[i].toString().length() - 6))) {
                        denumeratedSubreddit = i;
                        break;
                    }
                }
                for (int i = 0; i < allTitles.get(s).size(); i++) {
                    String answer;
                    try {
                        answer = rc.testData(allTitles.get(s).get(i));
                    } catch (Exception e) {
                        throw new TestDocException();
                    }
                    if (answer == null) {
                        throw new TestDocException("Null testData return value");
                    }
                    if (s.equals(answer)) {
                        System.out.println("Success!\tActual:" + s + "\t\t" + "Classified: " + answer);
                        //bufferedWriter.write("Success!\tActual:" + s + "\t\t" + "Classified: " + answer + "\n");
                        success++;
                        confusionMatrix[denumeratedSubreddit][denumeratedSubreddit]++;
                    } else {
                        System.out.println("Fail!\t\tActual:" + s + "\t\t" + "Classified: " + answer);
                        //bufferedWriter.write("Fail!\t\tActual:" + s + "\t\t" + "Classified: " + answer + "\n");
                        fail++;
                        int answerDenumeration = 0;
                        for (int j = 0; j < filenames.length; j++) {
                            if (answer.equals(filenames[j].toString().substring(21, filenames[j].toString().length() - 6))) {
                                answerDenumeration = j;
                                break;
                            }
                        }
                        confusionMatrix[denumeratedSubreddit][answerDenumeration]++;
                    }
                }
            }
            double accuracy = success * 1.0 / (success + fail);
            System.out.println("\nAccuracy: " + accuracy + "\n");
            //bufferedWriter.write("\nAccuracy: " + String.valueOf(accuracy) + "\n\n");
            int tp;
            int total;
            double precision;
            double recall;
            for (int i = 0; i < confusionMatrix.length; i++) {
                total = 0;
                for (int j = 0; j < confusionMatrix[i].length; j++) {
                    total += confusionMatrix[i][j];
                }
                tp = confusionMatrix[i][i];
                precision = (double) tp / total;
                System.out.println(filenames[i].toString().substring(21, filenames[i].toString().length() - 6));
                //  bufferedWriter.write(filenames[i].toString().substring(16, filenames[i].toString().length() - 6) + "\n");
                System.out.println("Precision:" + precision);
                //  bufferedWriter.write("Precision:" + precision + "\n");
                total = 0;
                for (int[] aConfusionMatrix : confusionMatrix) {
                    total += aConfusionMatrix[i];
                }
                recall = (double) tp / total;
                System.out.println("Recall" + " " + recall + "\n");
                //bufferedWriter.write("Recall" + " " + recall + "\n\n");
            }
            //bufferedWriter.close();

            System.out.println("CONFUSION MATRIX: ");
            for (int[] ints : confusionMatrix) {
                for (int anInt : ints) {
                    System.out.print(anInt + "\t");
                }
                System.out.println();
            }
            BufferedReader bufferedReader = new BufferedReader(new FileReader(referenceFile));
            double referenceAccuracy = Double.valueOf(bufferedReader.readLine());
            double grade = (accuracy / referenceAccuracy) * 100;
            System.out.println("\nYOUR GRADE IS: " + grade + "%");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (TrainingDocException e) {
            System.out.println("\n\n\n" + e.getMessage());
            System.out.println("Exception occurred in addTrainingDocument method of your NaiveBayesClassifier.java");
            System.out.println("\nYOUR GRADE IS: " + 0 + "%");
        } catch (TestDocException e) {
            System.out.println("\n\n\n" + e.getMessage());
            System.out.println("Exception occurred in testData method of your NaiveBayesClassifier.java");
            System.out.println("\nYOUR GRADE IS: " + 0 + "%");
        } catch (NullPointerException e) {
            System.out.println("Null Pointer Exception.!");
            System.out.println("\nYOUR GRADE IS: " + 0 + "%");
        }

    }
}