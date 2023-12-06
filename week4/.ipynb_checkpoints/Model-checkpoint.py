{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "99d65b6a-a5e9-42f2-8216-ce86cda73a98",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "ff660a2d-0853-44d8-b98a-ef851c7ca24b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Sepal_Length  Sepal_Width  Petal_Length  Petal_Width   Class\n",
      "0           5.1          3.5           1.4          0.2  Setosa\n",
      "1           4.9          3.0           1.4          0.2  Setosa\n",
      "2           4.7          3.2           1.3          0.2  Setosa\n",
      "3           4.6          3.1           1.5          0.2  Setosa\n",
      "4           5.0          3.6           1.4          0.2  Setosa\n"
     ]
    }
   ],
   "source": [
    "\n",
    "df_iris = pd.read_csv(\"iris.csv\")\n",
    "\n",
    "print(df_iris.head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d3ad7ca6-182a-4bb7-b058-d4f49f098eba",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[[\"Sepal_Length\", \"Sepal_Width\", \"Petal_Length\", \"Petal_Width\"]]\n",
    "y = df[\"Class\"]\n",
    "\n",
    "# Split the dataset into train and test\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "87ab717a-f195-454f-80c3-c6e715d0713d",
   "metadata": {},
   "outputs": [],
   "source": [
    "sc = StandardScaler()\n",
    "X_train = sc.fit_transform(X_train)\n",
    "X_test= sc.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "980a1a9e-b063-484b-9f7a-790212d73542",
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier = RandomForestClassifier()\n",
    "\n",
    "classifier.fit(X_train, y_train)\n",
    "\n",
    "pickle.dump(classifier, open(\"model.pkl\", \"wb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9fad722a-0f64-4948-b71d-3fe5437faaee",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
