import java.util.ArrayList;
import java.util.Map;
import java.util.TreeMap;

public class NaiveBayesClassifier implements NaiveBayesInterface {
    //dictionary maps each unique string to its document frequency
    //doc frequency is the number of documents it has appeared in
    private TreeMap<String, Integer> docFrequency;

    //data maps each string to a category, eg: "AskReddit" string
    //will be mapped to AskReddit category's documents info 
    private TreeMap<String, Category> data;

    //total number of documents added to the classifier
    private int totalDocumentCount;

    //words to ignore 
    public NaiveBayesClassifier() {
        docFrequency = new TreeMap<>();
        data = new TreeMap<>();
        totalDocumentCount = 0;
    }

    //given an input training doc, this method
    //converts string document to lowercase, removes punctuations, stems the words
    //and iterates through each individual string of the doc (delimited by whitespace)
    //and updates the fields of the classifier accordingly
    public void addTrainingDocument(String doc, String cat) {
        //increment the total document count
        totalDocumentCount++;

        //add this category to the data field if it doesn't exist 
        if (!data.containsKey(cat))
            data.put(cat, new Category());

        Category currentCat = data.get(cat);

        //increase the number of documents in this category
        currentCat.documentsInCategory++;

        Map<String, Integer> docTF = new TreeMap<>(); // will be added to category's documentTF arraylist
        //use clean to remove punctuation, switch to lowercase, stem, and split
        //the document on whitespace
        String[] documents = clean(doc);
        for (String current : documents) {
            //increment the TF count of this string in the document's treemap representation
            //and also, if appropriate, increment the doc frequency counter of string
            if (!docTF.containsKey(current)) {
                if (!docFrequency.containsKey(current))
                    docFrequency.put(current, 1);
                else
                    docFrequency.put(current, docFrequency.get(current) + 1);
                docTF.put(current, 1);
            } else
                docTF.put(current, docTF.get(current) + 1);

            //increment the TF count of this string in the category's treemap representation
            if (!data.get(cat).categoryTF.containsKey(current))
                currentCat.categoryTF.put(current, 1);
            else
                currentCat.categoryTF.put(current, currentCat.categoryTF.get(current) + 1);


        }
        //add this document's treemap TF representation to the current category's documentTF arraylist
        currentCat.documentTF.add(docTF);

    }

    //cleans and stems each string, and gets rid of stop words
    public String[] clean(String doc) {
        String[] titles = doc.toLowerCase().replaceAll("[^\\w ]", "").split("\\s+");
        StopWords stop = new StopWords();
        ArrayList<String> holder = new ArrayList<>();
        for (int i = 0; i < titles.length; i++) {
            if (stop.isStopWord(titles[i])) {
                titles[i] = null;
                continue;
            }
            Stemmer s = new Stemmer();
            char[] word = titles[i].toCharArray();
            s.add(word, word.length);
            s.stem();
            holder.add(s.toString());
        }


        titles = holder.toArray(new String[holder.size()]);
        return titles;

    }

    public TreeMap<String, Integer> getDocFrequency() {
        return docFrequency;
    }

    public TreeMap<String, Category> getData() {
        return data;
    }

    public int getTotalDocumentCount() {
        return totalDocumentCount;
    }

    public String testData(String doc) {
        //clean, stem, and vectorize the input test data document
        String[] newDoc = clean(doc);
        TreeMap<String, Integer> testTF = new TreeMap<>();
        for (String aNewDoc : newDoc) {
            if (!testTF.containsKey(aNewDoc))
                testTF.put(aNewDoc, 1);
            else
                testTF.put(aNewDoc, testTF.get(aNewDoc) + 1);
        }
        TreeMap<String, Category> categories = getData();
        TreeMap<String, Double> probabilities = new TreeMap<>();
        int vocabSize = getDocFrequency().keySet().size();
        for (String s : categories.keySet()) {
            Category currentCat = categories.get(s);
            double accumulator = 0;
            double prior = (currentCat.documentsInCategory * 1.0) / (getTotalDocumentCount() * 1.0);
            accumulator += Math.log10(prior);
            int totalWordsinCat = 0;
            for (String z : currentCat.categoryTF.keySet()) {
                totalWordsinCat += currentCat.categoryTF.get(z);
            }
            for (String z : testTF.keySet()) {
                int categoryOccurrences = 0;
                if (currentCat.categoryTF.keySet().contains(z))
                    categoryOccurrences = currentCat.categoryTF.get(z) + 1;
                else
                    categoryOccurrences = 1;
                double toAdd = (categoryOccurrences * 1.0) / (totalWordsinCat + vocabSize);
                toAdd = Math.log10(toAdd);
                accumulator += toAdd;
            }
            probabilities.put(s, accumulator);
        }
        String answer = "";
        double max = Double.NEGATIVE_INFINITY;
        for (String z : probabilities.keySet()) {
            if (probabilities.get(z) > max) {
                max = probabilities.get(z);
                answer = z;
            }
        }
        return answer;
    }
}