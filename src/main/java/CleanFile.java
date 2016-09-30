import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Victor Kwak, 9/30/16
 */
public class CleanFile {
    public static void main(String[] args) {
        List<String> sentences = new ArrayList<>();
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader("/Users/victorkwak/Documents/Repo/NaiveBayesReddit/Data/Debate/debate"))) {
            String current;
            String turn = "";
            while ((current = bufferedReader.readLine()) != null) {
                if (current.matches("\\s*")) {
                    continue;
                }
                if (current.matches("(CLINTON:|TRUMP:|HOLT:)\\s.*")) {
                    turn = current.split(":")[0];
                    sentences.add(current);
                } else {
                    sentences.add(turn + ": " + current);
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("/Users/victorkwak/Documents/Repo/NaiveBayesReddit/Data/Debate/debateClean"));
            for (String sentence : sentences) {
                bufferedWriter.write(sentence);
                bufferedWriter.newLine();
            }
            bufferedWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
