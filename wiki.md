# Usługa dokonująca analizy sentymentu
[1. Opis](#1.-opis)  
[2. Zródła](#2.-źródła)  
[3. Przykładowe dane](#3.-Przykładowe-dane)  
[4. Opis wywołania](#4.-Opis-wywołania)  
[5. Uruchomienie kontenera z usługą](#5.-Uruchomienie-kontenera-z-usługą)  
[6. Lokalizacja plików przykładowych](#6.-Lokalizacja-plików-przykładowych)

## 1. Opis
Usługa dokonuje oceny sentymentu na podstawie anglojęzycznego tekstu recenzji filmu.

Algorytm analizy dokonuje predykcji z użyciem sekwencyjnej sieci neuronowej o następującej strukturze:
1. Wejście - wektoryzacja ciągu znaków - przypisanie każdemu słowu unikalnej liczby całkowitej
2. Dropout - wyzerowanie losowych wejść w celu uniknięcia overfittingu
3. Embedding - powiązanie każdego słowa z wektorem o stałej długości
4. Global Average Pooling 1D - zwraca wektor o stałej długości za pomocą uśredniania, co pozwala na
proste przetwarzanie tekstów dowolnej długości
5. Dropout - jak w 2.
6. Dense - gęsty neuron zwracający pojedynczą wartość
7. Wyjście - sigmoidalna funkcja aktywacji

Jednocześnie model jest w stanie wskazać słowa wywierające największy wpływ na ostateczną predykcję,
wybierając największą wagę wśród poszczególnych wejść neuronu 6., a następnie
wybierając te słowa, które mają największą/najmniejszą wartość na tej samej pozycji, co wcześniej wyznaczona waga.

## 2. Źródła

Model klasyfikacyjny został utworzony z wykorzystaniem następującego przewodnika:
[Basic text classification](https://www.tensorflow.org/tutorials/keras/text_classification)

Dane treningowe pochodzą ze zbioru Large Movie Review Dataset:
[Large Movie Review Dataset](https://ai.stanford.edu/%7Eamaas/data/sentiment/)

## 3. Przykładowe dane
Przykład 1. Recenzja pozytywna:
```json
{
  "review": "The film was good. If I was asked to watch it again, I would watch it. Maybe I won't say it was good. In fact, it was brilliant."
}
```

Przykład 2. Recenzja negatywna:
```json
{
  "review": "If anyone said I should see this film, I wouldn't hide my disgust. It was a terrible experience."
}
```

Oczekiwany format danych to obiekt JSON z kluczem `"review"` i wartością będącą ciągiem znaków - recenzją.

## 4. Opis wywołania
Po uruchomieniu usługi można sprawdzić jej działanie korzystając z programu np. Postman.

1.    Ścieżka do sprawdzenia statusu usługi: `/status`
    
      Zwraca: `{"version": 0.2, "running": true}`

2.    Ścieżka do algorytmu: `/predict`
    
      Ścieżka zwraca 3 wartości: `description` - słowny opis predykcji, 
      `mostInfluentialVars` - słowa, które wywarły największy wpływ na predykcję, `result` -
      liczbowa wartość predykcji z zakresu [0; 1].
      ```json
      {
        "description": "The review is positive.",
        "mostInfulentialVars": [
            "brilliant",
            "good",
            "watch",
            "it",
            "again"
        ],
        "result": 0.7003768682479858
      }
      ```

## 5. Uruchomienie kontenera z usługą

W celu uruchomienia kontenera można posłużyć się plikiem Dockerfile dołączonym do projektu:
```
docker build -t text-classification-container-name .
docker run -p 5001:5001 text-classification-container-name
```
Można też kontener pobrać i uruchomić bezpośrednio z Docker Hub:
```
docker run -p 5001:5001 staszekm/text-classification
```

Powyższa konfiguracja pozwala na komunikację z usługą przez port 5001, np.
`http://localhost:5001/status`

## 6. Lokalizacja plików przykładowych
Pliki przykładowe w formacie .json są dostępne w folderze ./data.